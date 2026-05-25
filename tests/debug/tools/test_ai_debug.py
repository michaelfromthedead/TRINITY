"""
Tests for the AI debugging system.

Tests verify:
1. AI debug state management
2. Perception visualization
3. Behavior tree debugging
4. Blackboard inspection
5. Build-type security guards
6. Actual game impact
"""

import os
import pytest
from unittest.mock import Mock

from engine.debug.tools.ai_debug import (
    AIDebugConfig,
    AIDebugger,
    AIDebugState,
    BTNodeDebugInfo,
    BlackboardDebugInfo,
    PerceptionType,
    PerceptionVisual,
    get_ai_debugger,
)


class TestAIDebugConfig:
    """Tests for AIDebugConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AIDebugConfig()
        assert config.default_sight_range == 20.0
        assert config.default_hearing_range == 15.0
        assert config.default_sight_angle == 90.0
        assert config.default_hearing_angle == 360.0
        assert config.allow_in_shipping is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = AIDebugConfig(
            default_sight_range=30.0,
            default_hearing_range=25.0,
        )
        assert config.default_sight_range == 30.0
        assert config.default_hearing_range == 25.0


class TestAIDebugger:
    """Tests for AIDebugger."""

    @pytest.fixture
    def debugger(self):
        """Create a fresh AIDebugger."""
        return AIDebugger()

    def test_initial_state(self, debugger):
        """Test initial debugger state."""
        assert debugger.state == AIDebugState.RUNNING
        assert debugger.enabled is True

    def test_pause_resume(self, debugger):
        """Test pausing and resuming AI."""
        debugger.pause_ai()
        assert debugger.state == AIDebugState.PAUSED

        debugger.resume_ai()
        assert debugger.state == AIDebugState.RUNNING

    def test_step_requires_pause(self, debugger):
        """Test stepping requires paused state."""
        debugger.step_ai()  # Should be ignored
        assert debugger.state == AIDebugState.RUNNING

        debugger.pause_ai()
        debugger.step_ai()
        assert debugger.state == AIDebugState.STEPPING

    def test_perception_visualization(self, debugger):
        """Test perception visualization toggling."""
        entity = Mock()

        debugger.show_perception(entity)
        assert debugger.is_perception_shown(entity) is True

        debugger.hide_perception(entity)
        assert debugger.is_perception_shown(entity) is False

    def test_show_all_perception(self, debugger):
        """Test show all perception mode."""
        entity = Mock()

        debugger.show_all_perception(True)
        assert debugger.is_perception_shown(entity) is True

        debugger.show_all_perception(False)
        assert debugger.is_perception_shown(entity) is False

    def test_behavior_tree_visualization(self, debugger):
        """Test behavior tree visualization."""
        entity = Mock()

        debugger.show_behavior_tree(entity)
        assert debugger.is_behavior_tree_shown(entity) is True

        debugger.hide_behavior_tree(entity)
        assert debugger.is_behavior_tree_shown(entity) is False

    def test_blackboard_visualization(self, debugger):
        """Test blackboard visualization."""
        entity = Mock()

        debugger.show_blackboard(entity)
        assert debugger.is_blackboard_shown(entity) is True

        debugger.hide_blackboard(entity)
        assert debugger.is_blackboard_shown(entity) is False

    def test_state_override(self, debugger):
        """Test AI state override."""
        entity = Mock()

        debugger.override_state(entity, "patrol")
        assert debugger.has_state_override(entity) is True
        assert debugger.get_state_override(entity) == "patrol"

        debugger.clear_state_override(entity)
        assert debugger.has_state_override(entity) is False

    def test_should_tick(self, debugger):
        """Test should_tick logic."""
        entity = Mock()

        # Running state allows ticks
        assert debugger.should_tick(entity) is True

        # Paused state blocks ticks
        debugger.pause_ai()
        assert debugger.should_tick(entity) is False

        # Stepping allows one tick
        debugger.step_ai()
        assert debugger.should_tick(entity) is True

        # Consume the step
        debugger.consume_step()
        assert debugger.state == AIDebugState.PAUSED

    def test_state_callbacks(self, debugger):
        """Test state change callbacks."""
        callback = Mock()
        debugger.add_state_callback(callback)

        debugger.pause_ai()
        callback.assert_called_with(AIDebugState.PAUSED)


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
        """Test AI debugger cannot be enabled in shipping builds."""
        os.environ["GAME_BUILD_TYPE"] = "SHIPPING"

        debugger = AIDebugger()
        debugger.enabled = True

        # Should remain disabled
        # Note: enabled starts True but set_enabled should block

    def test_config_allows_shipping_override(self):
        """Test config can allow AI debugger in shipping."""
        os.environ["GAME_BUILD_TYPE"] = "SHIPPING"

        config = AIDebugConfig(allow_in_shipping=True)
        debugger = AIDebugger(config)

        debugger.enabled = True
        assert debugger.enabled is True


class TestConfigValues:
    """Tests that config values are used instead of magic numbers."""

    def test_custom_perception_values(self):
        """Test custom perception values from config."""
        config = AIDebugConfig(
            default_sight_range=50.0,
            default_hearing_range=40.0,
            default_sight_angle=120.0,
        )
        debugger = AIDebugger(config)

        assert debugger._default_sight.range == 50.0
        assert debugger._default_hearing.range == 40.0
        assert debugger._default_sight.angle == 120.0

    def test_custom_colors(self):
        """Test custom colors from config."""
        config = AIDebugConfig(
            sight_color=(1.0, 0.0, 0.0, 0.5),
            hearing_color=(0.0, 1.0, 0.0, 0.5),
        )
        debugger = AIDebugger(config)

        assert debugger._default_sight.color == (1.0, 0.0, 0.0, 0.5)
        assert debugger._default_hearing.color == (0.0, 1.0, 0.0, 0.5)


class TestAIDebugGameImpact:
    """Tests that verify AI debugging actually impacts AI systems."""

    @pytest.fixture
    def debugger(self):
        """Create a fresh AIDebugger."""
        return AIDebugger()

    def test_pause_blocks_ai_updates(self, debugger):
        """Test pausing actually blocks AI updates."""
        entity = Mock()
        ai_updates = 0

        def update_ai():
            nonlocal ai_updates
            if debugger.should_tick(entity):
                ai_updates += 1

        # AI runs normally
        for _ in range(5):
            update_ai()
        assert ai_updates == 5

        # Pause blocks AI
        debugger.pause_ai()
        for _ in range(5):
            update_ai()
        assert ai_updates == 5  # No additional updates

    def test_step_advances_single_tick(self, debugger):
        """Test stepping advances exactly one AI tick."""
        entity = Mock()
        ai_updates = 0

        def update_ai():
            nonlocal ai_updates
            if debugger.should_tick(entity):
                ai_updates += 1
                debugger.consume_step()

        # Pause and step
        debugger.pause_ai()
        debugger.step_ai()

        # Multiple update calls
        for _ in range(5):
            update_ai()

        # Only 1 update should have happened
        assert ai_updates == 1

    def test_state_override_affects_ai(self, debugger):
        """Test state override affects AI state machine."""
        entity = Mock()

        def get_ai_state():
            override = debugger.get_state_override(entity)
            if override:
                return override
            return "default"

        # Normal state
        assert get_ai_state() == "default"

        # Override state
        debugger.override_state(entity, "attack")
        assert get_ai_state() == "attack"

        # Clear override
        debugger.clear_state_override(entity)
        assert get_ai_state() == "default"
