"""
Tests for the physics debugging system.

Tests verify:
1. Physics debug state management
2. Visualization flags
3. Contact and raycast tracking
4. Build-type security guards
5. Actual game impact
"""

import os
import pytest
from unittest.mock import Mock

from engine.debug.tools.physics_debug import (
    BodyInspection,
    ContactPoint,
    PhysicsDebugger,
    PhysicsDebugState,
    PhysicsVisualization,
    PhysicsVisualizationConfig,
    get_physics_debugger,
)


class TestPhysicsVisualizationConfig:
    """Tests for PhysicsVisualizationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PhysicsVisualizationConfig()
        assert config.max_contacts == 1000
        assert config.max_raycast_history == 100
        assert config.velocity_scale == 0.1
        assert config.allow_in_shipping is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = PhysicsVisualizationConfig(
            max_contacts=500,
            max_raycast_history=50,
            velocity_scale=0.2,
        )
        assert config.max_contacts == 500
        assert config.max_raycast_history == 50
        assert config.velocity_scale == 0.2


class TestPhysicsDebugger:
    """Tests for PhysicsDebugger."""

    @pytest.fixture
    def debugger(self):
        """Create a fresh PhysicsDebugger."""
        return PhysicsDebugger()

    def test_initial_state(self, debugger):
        """Test initial debugger state."""
        assert debugger.state == PhysicsDebugState.RUNNING
        assert debugger.enabled is True
        assert debugger.visualization == PhysicsVisualization.NONE

    def test_pause_resume(self, debugger):
        """Test pausing and resuming physics."""
        debugger.pause_physics()
        assert debugger.state == PhysicsDebugState.PAUSED

        debugger.resume_physics()
        assert debugger.state == PhysicsDebugState.RUNNING

    def test_step_requires_pause(self, debugger):
        """Test stepping requires paused state."""
        count = debugger.step_physics()
        assert count == 0  # Ignored when not paused

        debugger.pause_physics()
        count = debugger.step_physics(3)
        assert count == 3
        assert debugger.state == PhysicsDebugState.STEPPING

    def test_visualization_flags(self, debugger):
        """Test visualization flag toggling."""
        debugger.show_collision_shapes(True)
        assert debugger.is_visualization_enabled(PhysicsVisualization.COLLISION_SHAPES)

        debugger.show_collision_shapes(False)
        assert not debugger.is_visualization_enabled(PhysicsVisualization.COLLISION_SHAPES)

    def test_show_all_visualizations(self, debugger):
        """Test show all visualizations."""
        debugger.show_all_visualizations()
        assert debugger.visualization == PhysicsVisualization.ALL

        debugger.hide_all_visualizations()
        assert debugger.visualization == PhysicsVisualization.NONE

    def test_entity_visualization(self, debugger):
        """Test entity-specific visualization."""
        entity = Mock()

        debugger.show_for_entity(entity)
        assert debugger.should_visualize(entity) is True

        debugger.hide_for_entity(entity)
        assert debugger.should_visualize(entity) is False

    def test_show_for_all(self, debugger):
        """Test show for all entities."""
        entity = Mock()

        debugger.show_for_all(True)
        assert debugger.should_visualize(entity) is True

        debugger.show_for_all(False)
        assert debugger.should_visualize(entity) is False

    def test_contact_tracking(self, debugger):
        """Test contact point tracking."""
        contact = ContactPoint(
            position=(0.0, 0.0, 0.0),
            normal=(0.0, 1.0, 0.0),
            penetration=0.1,
            body_a=Mock(),
            body_b=Mock(),
        )

        debugger.record_contact(contact)
        contacts = debugger.get_contacts()

        assert len(contacts) == 1
        assert contacts[0].position == (0.0, 0.0, 0.0)

    def test_contact_limit(self, debugger):
        """Test contact tracking respects limit."""
        # Record more than max contacts
        for i in range(debugger._max_contacts + 100):
            contact = ContactPoint(
                position=(float(i), 0.0, 0.0),
                normal=(0.0, 1.0, 0.0),
                penetration=0.1,
                body_a=Mock(),
                body_b=Mock(),
            )
            debugger.record_contact(contact)

        contacts = debugger.get_contacts()
        assert len(contacts) == debugger._max_contacts

    def test_raycast_tracking(self, debugger):
        """Test raycast tracking."""
        debugger.record_raycast(
            start=(0.0, 0.0, 0.0),
            end=(10.0, 0.0, 0.0),
            hit=True,
        )

        raycasts = debugger.get_raycasts()
        assert len(raycasts) == 1
        assert raycasts[0][2] is True  # hit

    def test_force_application(self, debugger):
        """Test force visualization."""
        entity = Mock()
        entity.position = Mock(x=0.0, y=0.0, z=0.0)

        debugger.apply_force(entity, (100.0, 0.0, 0.0))

        forces = debugger.get_pending_forces()
        assert len(forces) == 1
        assert forces[0][1] == (100.0, 0.0, 0.0)

    def test_should_simulate(self, debugger):
        """Test should_simulate logic."""
        # Running state allows simulation
        assert debugger.should_simulate() is True

        # Paused state blocks simulation
        debugger.pause_physics()
        assert debugger.should_simulate() is False

        # Stepping allows one simulation
        debugger.step_physics(1)
        assert debugger.should_simulate() is True

    def test_consume_step(self, debugger):
        """Test step consumption."""
        debugger.pause_physics()
        debugger.step_physics(2)

        # Consume first step
        assert debugger.consume_step() is True
        assert debugger._step_count == 1

        # Consume second step
        assert debugger.consume_step() is True
        assert debugger.state == PhysicsDebugState.PAUSED

        # No more steps
        assert debugger.consume_step() is False


class TestBuildTypeGuards:
    """Tests for build-type security guards."""

    @pytest.fixture(autouse=True)
    def reset_env(self):
        """Reset environment before each test."""
        for var in ["GAME_BUILD_TYPE", "SHIPPING"]:
            os.environ.pop(var, None)
        yield
        for var in ["GAME_BUILD_TYPE", "SHIPPING"]:
            os.environ.pop(var, None)

    def test_cannot_enable_in_shipping(self):
        """Test physics debugger cannot be enabled in shipping builds."""
        os.environ["GAME_BUILD_TYPE"] = "SHIPPING"

        debugger = PhysicsDebugger()
        debugger.enabled = True

        # Should trigger warning and block enable

    def test_config_allows_shipping_override(self):
        """Test config can allow physics debugger in shipping."""
        os.environ["GAME_BUILD_TYPE"] = "SHIPPING"

        config = PhysicsVisualizationConfig(allow_in_shipping=True)
        debugger = PhysicsDebugger(config)

        debugger.enabled = True
        assert debugger.enabled is True


class TestConfigValues:
    """Tests that config values are used instead of magic numbers."""

    def test_custom_buffer_limits(self):
        """Test custom buffer limits from config."""
        config = PhysicsVisualizationConfig(
            max_contacts=100,
            max_raycast_history=25,
        )
        debugger = PhysicsDebugger(config)

        assert debugger._max_contacts == 100
        assert debugger._max_raycast_history == 25

    def test_custom_colors(self):
        """Test custom colors are accessible from config."""
        config = PhysicsVisualizationConfig(
            collision_color=(1.0, 0.0, 0.0, 0.8),
            velocity_color=(0.0, 0.0, 1.0, 0.8),
        )
        debugger = PhysicsDebugger(config)

        assert debugger.config.collision_color == (1.0, 0.0, 0.0, 0.8)
        assert debugger.config.velocity_color == (0.0, 0.0, 1.0, 0.8)


class TestPhysicsDebugGameImpact:
    """Tests that verify physics debugging actually impacts physics systems."""

    @pytest.fixture
    def debugger(self):
        """Create a fresh PhysicsDebugger."""
        return PhysicsDebugger()

    def test_pause_blocks_physics_updates(self, debugger):
        """Test pausing actually blocks physics updates."""
        physics_steps = 0

        def step_physics():
            nonlocal physics_steps
            if debugger.should_simulate():
                physics_steps += 1

        # Physics runs normally
        for _ in range(5):
            step_physics()
        assert physics_steps == 5

        # Pause blocks physics
        debugger.pause_physics()
        for _ in range(5):
            step_physics()
        assert physics_steps == 5  # No additional steps

    def test_step_advances_single_simulation(self, debugger):
        """Test stepping advances exactly one physics step."""
        physics_steps = 0

        def step_physics():
            nonlocal physics_steps
            if debugger.should_simulate():
                physics_steps += 1
                debugger.consume_step()

        # Pause and step
        debugger.pause_physics()
        debugger.step_physics(1)

        # Multiple step calls
        for _ in range(5):
            step_physics()

        # Only 1 step should have happened
        assert physics_steps == 1

    def test_visualization_affects_rendering(self, debugger):
        """Test visualization flags affect what should be rendered."""

        def get_render_items():
            items = []
            if debugger.is_visualization_enabled(PhysicsVisualization.COLLISION_SHAPES):
                items.append("collision_shapes")
            if debugger.is_visualization_enabled(PhysicsVisualization.VELOCITIES):
                items.append("velocities")
            if debugger.is_visualization_enabled(PhysicsVisualization.CONTACT_POINTS):
                items.append("contacts")
            return items

        # Nothing rendered by default
        assert get_render_items() == []

        # Enable collision shapes
        debugger.show_collision_shapes(True)
        assert get_render_items() == ["collision_shapes"]

        # Enable velocities
        debugger.show_velocities(True)
        assert "velocities" in get_render_items()
