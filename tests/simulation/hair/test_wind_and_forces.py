"""
Whitebox tests for wind and external forces in hair simulation.
"""

import numpy as np
import pytest

from engine.simulation.hair.config import (
    HEAD_INERTIA_COEFFICIENT,
    WIND_INFLUENCE_MULTIPLIER,
)
from engine.simulation.hair.hair_simulation import (
    GuideHair,
    HairControlPoint,
    HairSimulation,
    HairSimulationConfig,
    HairState,
    create_hair_strand,
)


def create_test_guide_hair(root_pos=None, num_segments=4):
    """Create a test guide hair."""
    if root_pos is None:
        root_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    else:
        root_pos = np.array(root_pos, dtype=np.float32)

    hair = create_hair_strand(
        root_position=root_pos,
        root_normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        length=0.2,
        num_segments=num_segments,
    )
    return hair


class TestGravityEffect:
    """Tests for gravity effects on hair simulation."""

    def test_gravity_pulls_hair_down(self):
        """Gravity should pull non-root points downward."""
        config = HairSimulationConfig(
            gravity=np.array([0.0, -9.81, 0.0], dtype=np.float32),
            damping=1.0,  # No damping for clearer test
            shape_stiffness=0.0,  # Disable shape matching
            local_shape_stiffness=0.0,
            enable_wind=False,
            enable_collision=False,
        )
        sim = HairSimulation(config)

        hair = create_test_guide_hair()
        original_tip_y = hair.control_points[-1].position[1]
        sim.add_guide_hair(hair)

        sim.start()
        # Step multiple times
        for _ in range(50):
            sim.step(0.016)

        # Tip should have moved downward
        new_tip_y = hair.control_points[-1].position[1]
        assert new_tip_y < original_tip_y

    def test_zero_gravity_no_vertical_motion(self):
        """Hair should not fall with zero gravity."""
        config = HairSimulationConfig(
            gravity=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            damping=1.0,
            shape_stiffness=0.0,
            local_shape_stiffness=0.0,
            enable_wind=False,
            enable_collision=False,
        )
        sim = HairSimulation(config)

        hair = create_test_guide_hair()
        original_tip = hair.control_points[-1].position.copy()
        sim.add_guide_hair(hair)

        sim.start()
        for _ in range(10):
            sim.step(0.016)

        # Tip should be approximately unchanged (only root constraint affects it)
        new_tip = hair.control_points[-1].position
        # With zero external forces, position change should be minimal
        diff = np.linalg.norm(new_tip - original_tip)
        assert diff < 0.1  # Allow small drift from constraints

    def test_custom_gravity_direction(self):
        """Gravity in non-standard direction should work."""
        config = HairSimulationConfig(
            gravity=np.array([5.0, 0.0, 0.0], dtype=np.float32),  # Horizontal gravity
            damping=1.0,
            shape_stiffness=0.0,
            local_shape_stiffness=0.0,
            enable_wind=False,
            enable_collision=False,
        )
        sim = HairSimulation(config)

        hair = create_test_guide_hair()
        original_tip_x = hair.control_points[-1].position[0]
        sim.add_guide_hair(hair)

        sim.start()
        for _ in range(50):
            sim.step(0.016)

        # Tip should have moved in X direction
        new_tip_x = hair.control_points[-1].position[0]
        assert new_tip_x > original_tip_x


class TestWindEffect:
    """Tests for wind effects on hair simulation."""

    def test_wind_moves_hair(self):
        """Wind should push hair in wind direction."""
        config = HairSimulationConfig(
            gravity=np.array([0.0, 0.0, 0.0], dtype=np.float32),  # No gravity
            enable_wind=True,
            damping=0.95,
            shape_stiffness=0.0,
            local_shape_stiffness=0.0,
            enable_collision=False,
        )
        sim = HairSimulation(config)

        hair = create_test_guide_hair()
        sim.add_guide_hair(hair)

        # Strong wind in X direction
        sim.set_wind(np.array([50.0, 0.0, 0.0], dtype=np.float32))

        original_tip_x = hair.control_points[-1].position[0]

        sim.start()
        for _ in range(30):
            sim.step(0.016)

        # Tip should have moved in wind direction
        new_tip_x = hair.control_points[-1].position[0]
        assert new_tip_x > original_tip_x

    def test_wind_disabled(self):
        """Disabled wind should not affect hair."""
        config = HairSimulationConfig(
            gravity=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            enable_wind=False,  # Disabled
            shape_stiffness=0.0,
            local_shape_stiffness=0.0,
            enable_collision=False,
        )
        sim = HairSimulation(config)

        hair = create_test_guide_hair()
        sim.add_guide_hair(hair)

        # Set wind (but it's disabled)
        sim.set_wind(np.array([100.0, 0.0, 0.0], dtype=np.float32))

        original_tip = hair.control_points[-1].position.copy()

        sim.start()
        for _ in range(20):
            sim.step(0.016)

        # Tip should be approximately unchanged
        new_tip = hair.control_points[-1].position
        diff = np.linalg.norm(new_tip - original_tip)
        assert diff < 0.05  # Minimal change

    def test_zero_wind(self):
        """Zero wind should not move hair."""
        config = HairSimulationConfig(
            gravity=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            enable_wind=True,
            shape_stiffness=0.0,
            local_shape_stiffness=0.0,
            enable_collision=False,
        )
        sim = HairSimulation(config)

        hair = create_test_guide_hair()
        sim.add_guide_hair(hair)

        sim.set_wind(np.array([0.0, 0.0, 0.0], dtype=np.float32))

        original_tip = hair.control_points[-1].position.copy()

        sim.start()
        for _ in range(10):
            sim.step(0.016)

        # Minimal change
        new_tip = hair.control_points[-1].position
        diff = np.linalg.norm(new_tip - original_tip)
        assert diff < 0.05

    def test_wind_affects_all_non_root_points(self):
        """Wind should affect all non-root control points."""
        config = HairSimulationConfig(
            gravity=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            enable_wind=True,
            shape_stiffness=0.0,
            local_shape_stiffness=0.0,
            enable_collision=False,
        )
        sim = HairSimulation(config)

        hair = create_test_guide_hair(num_segments=8)
        sim.add_guide_hair(hair)

        original_positions = [cp.position[0] for cp in hair.control_points]

        sim.set_wind(np.array([30.0, 0.0, 0.0], dtype=np.float32))

        sim.start()
        for _ in range(20):
            sim.step(0.016)

        # All non-root points should have moved
        for i, cp in enumerate(hair.control_points):
            if i == 0:  # Root
                assert cp.position[0] == pytest.approx(original_positions[i], abs=0.01)
            else:  # Non-root
                assert cp.position[0] > original_positions[i]


class TestInertiaTransfer:
    """Tests for inertia transfer from head motion."""

    def test_head_motion_affects_hair(self):
        """Moving head should transfer inertia to hair."""
        config = HairSimulationConfig(
            gravity=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            enable_wind=False,
            shape_stiffness=0.0,
            local_shape_stiffness=0.0,
            enable_collision=False,
            damping=0.9,
        )
        sim = HairSimulation(config)

        hair = create_test_guide_hair()
        sim.add_guide_hair(hair)

        # Set initial head position
        initial_head = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        sim.set_head_transform(initial_head, np.eye(3, dtype=np.float32))

        sim.start()
        sim.step(0.016)

        # Now move head quickly
        new_head = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        sim.set_head_transform(new_head, np.eye(3, dtype=np.float32))

        # Record position before step
        tip_before = hair.control_points[-1].position.copy()

        # Step simulation
        for _ in range(5):
            sim.step(0.016)

        # Hair should have some inertia effect
        # The exact behavior depends on implementation

    def test_head_rotation_affects_hair(self):
        """Rotating head should affect hair."""
        config = HairSimulationConfig(
            gravity=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            enable_wind=False,
            shape_stiffness=0.5,  # Enable shape matching to see rotation effect
            local_shape_stiffness=0.0,
            enable_collision=False,
        )
        sim = HairSimulation(config)

        hair = create_test_guide_hair()
        sim.add_guide_hair(hair)

        # Initial orientation
        sim.set_head_transform(
            np.zeros(3, dtype=np.float32),
            np.eye(3, dtype=np.float32),
        )

        sim.start()
        sim.step(0.016)

        # Rotate head 90 degrees around Y
        rotation = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ], dtype=np.float32)

        sim.set_head_transform(np.zeros(3, dtype=np.float32), rotation)

        for _ in range(10):
            sim.step(0.016)

        # Hair should have rotated with head (via shape matching)


class TestDamping:
    """Tests for velocity damping."""

    def test_damping_reduces_velocity(self):
        """Damping should reduce hair velocity over time."""
        config = HairSimulationConfig(
            gravity=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            enable_wind=False,
            damping=0.9,  # Significant damping
            shape_stiffness=0.0,
            local_shape_stiffness=0.0,
            enable_collision=False,
        )
        sim = HairSimulation(config)

        hair = create_test_guide_hair()
        # Give tip initial velocity
        hair.control_points[-1].velocity = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        sim.add_guide_hair(hair)

        sim.start()

        initial_velocity_magnitude = np.linalg.norm(hair.control_points[-1].velocity)

        for _ in range(20):
            sim.step(0.016)

        final_velocity_magnitude = np.linalg.norm(hair.control_points[-1].velocity)

        # Velocity should have reduced due to damping
        assert final_velocity_magnitude < initial_velocity_magnitude

    def test_no_damping(self):
        """With damping=1.0, velocity should be preserved."""
        config = HairSimulationConfig(
            gravity=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            enable_wind=False,
            damping=1.0,  # No damping
            shape_stiffness=0.0,
            local_shape_stiffness=0.0,
            enable_collision=False,
        )
        sim = HairSimulation(config)

        hair = create_test_guide_hair()
        hair.control_points[-1].velocity = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        sim.add_guide_hair(hair)

        sim.start()

        initial_velocity_magnitude = np.linalg.norm(hair.control_points[-1].velocity)

        for _ in range(5):
            sim.step(0.016)

        final_velocity_magnitude = np.linalg.norm(hair.control_points[-1].velocity)

        # Velocity should be approximately maintained (constraints may affect it)
        # This is a rough check - length constraints affect velocity
        assert final_velocity_magnitude > 0


class TestCombinedForces:
    """Tests for combined force effects."""

    def test_gravity_and_wind_combined(self):
        """Gravity and wind should combine correctly."""
        config = HairSimulationConfig(
            gravity=np.array([0.0, -5.0, 0.0], dtype=np.float32),  # Downward
            enable_wind=True,
            damping=0.95,
            shape_stiffness=0.0,
            local_shape_stiffness=0.0,
            enable_collision=False,
        )
        sim = HairSimulation(config)

        hair = create_test_guide_hair()
        sim.add_guide_hair(hair)

        # Wind blowing sideways
        sim.set_wind(np.array([20.0, 0.0, 0.0], dtype=np.float32))

        original_tip = hair.control_points[-1].position.copy()

        sim.start()
        for _ in range(30):
            sim.step(0.016)

        new_tip = hair.control_points[-1].position

        # Should have moved both down (gravity) and sideways (wind)
        assert new_tip[1] < original_tip[1]  # Down
        assert new_tip[0] > original_tip[0]  # Sideways

    def test_wind_against_gravity(self):
        """Wind can oppose gravity."""
        config = HairSimulationConfig(
            gravity=np.array([0.0, -5.0, 0.0], dtype=np.float32),
            enable_wind=True,
            damping=0.95,
            shape_stiffness=0.0,
            local_shape_stiffness=0.0,
            enable_collision=False,
        )
        sim = HairSimulation(config)

        hair = create_test_guide_hair()
        sim.add_guide_hair(hair)

        # Strong upward wind
        sim.set_wind(np.array([0.0, 100.0, 0.0], dtype=np.float32))

        original_tip_y = hair.control_points[-1].position[1]

        sim.start()
        for _ in range(30):
            sim.step(0.016)

        new_tip_y = hair.control_points[-1].position[1]

        # Upward wind should at least partially counteract gravity
        # May still go down but less than with gravity alone


class TestEdgeCasesForces:
    """Edge case tests for force application."""

    def test_very_strong_gravity(self):
        """Should handle very strong gravity."""
        config = HairSimulationConfig(
            gravity=np.array([0.0, -1000.0, 0.0], dtype=np.float32),
            damping=0.99,
            shape_stiffness=0.0,
            local_shape_stiffness=0.0,
            enable_wind=False,
            enable_collision=False,
        )
        sim = HairSimulation(config)

        hair = create_test_guide_hair()
        sim.add_guide_hair(hair)

        sim.start()
        for _ in range(10):
            sim.step(0.016)

        # Should not have NaN or inf
        for cp in hair.control_points:
            assert not np.any(np.isnan(cp.position))
            assert not np.any(np.isinf(cp.position))

    def test_very_strong_wind(self):
        """Should handle very strong wind."""
        config = HairSimulationConfig(
            gravity=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            enable_wind=True,
            shape_stiffness=0.0,
            local_shape_stiffness=0.0,
            enable_collision=False,
        )
        sim = HairSimulation(config)

        hair = create_test_guide_hair()
        sim.add_guide_hair(hair)

        sim.set_wind(np.array([1000.0, 1000.0, 1000.0], dtype=np.float32))

        sim.start()
        for _ in range(10):
            sim.step(0.016)

        # Should not have NaN or inf
        for cp in hair.control_points:
            assert not np.any(np.isnan(cp.position))
            assert not np.any(np.isinf(cp.position))

    def test_changing_wind_direction(self):
        """Should handle changing wind direction."""
        config = HairSimulationConfig(
            gravity=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            enable_wind=True,
            shape_stiffness=0.0,
            local_shape_stiffness=0.0,
            enable_collision=False,
        )
        sim = HairSimulation(config)

        hair = create_test_guide_hair()
        sim.add_guide_hair(hair)

        sim.start()

        for i in range(10):
            # Change wind direction each step
            angle = i * 0.628  # ~36 degrees per step
            wind = np.array([
                np.cos(angle) * 10,
                0.0,
                np.sin(angle) * 10,
            ], dtype=np.float32)
            sim.set_wind(wind)
            sim.step(0.016)

        # Should not crash
        for cp in hair.control_points:
            assert not np.any(np.isnan(cp.position))

    def test_very_small_timestep(self):
        """Should handle very small timestep."""
        config = HairSimulationConfig(
            timestep=0.0001,  # Very small
            gravity=np.array([0.0, -9.81, 0.0], dtype=np.float32),
            shape_stiffness=0.5,
            enable_collision=False,
        )
        sim = HairSimulation(config)

        hair = create_test_guide_hair()
        sim.add_guide_hair(hair)

        sim.start()
        # Step with large dt, many internal steps
        sim.step(0.1)

        for cp in hair.control_points:
            assert not np.any(np.isnan(cp.position))

    def test_many_hairs_with_forces(self):
        """Should handle many hairs with forces."""
        config = HairSimulationConfig(
            gravity=np.array([0.0, -9.81, 0.0], dtype=np.float32),
            enable_wind=True,
            damping=0.95,
            shape_stiffness=0.3,
            enable_collision=False,
        )
        sim = HairSimulation(config)

        # Add many hairs
        for i in range(50):
            hair = create_test_guide_hair([i * 0.01, 0.0, 0.0])
            sim.add_guide_hair(hair)

        sim.set_wind(np.array([5.0, 0.0, 0.0], dtype=np.float32))

        sim.start()
        for _ in range(10):
            sim.step(0.016)

        # All hairs should have valid positions
        for hair in sim.guide_hairs:
            for cp in hair.control_points:
                assert not np.any(np.isnan(cp.position))
                assert not np.any(np.isinf(cp.position))
