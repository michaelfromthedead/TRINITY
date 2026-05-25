"""
Tests for the time control system.
"""

import pytest
from unittest.mock import Mock

from engine.debug.tools.time_control import (
    TimeController,
    TimeControlConfig,
    TimeState,
    get_time_controller,
)


class TestTimeControlConfig:
    """Tests for TimeControlConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TimeControlConfig()
        assert config.min_time_scale == 0.01
        assert config.max_time_scale == 10.0
        assert config.default_slow_motion == 0.25
        assert config.default_fast_forward == 2.0
        assert config.frame_step_scale == 1.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = TimeControlConfig(
            min_time_scale=0.1,
            max_time_scale=4.0,
            default_slow_motion=0.5,
        )
        assert config.min_time_scale == 0.1
        assert config.max_time_scale == 4.0
        assert config.default_slow_motion == 0.5


class TestTimeController:
    """Tests for TimeController."""

    @pytest.fixture
    def controller(self):
        """Create a fresh TimeController."""
        return TimeController()

    def test_initial_state(self, controller):
        """Test initial controller state."""
        assert controller.time_scale == 1.0
        assert controller.is_paused is False
        assert controller.state == TimeState.NORMAL
        assert controller.frame_step_pending is False

    def test_pause(self, controller):
        """Test pausing."""
        controller.pause()
        assert controller.is_paused is True
        assert controller.time_scale == 0.0
        assert controller.state == TimeState.PAUSED

    def test_pause_when_already_paused(self, controller):
        """Test pausing when already paused."""
        controller.pause()
        controller.pause()  # Should not change anything
        assert controller.is_paused is True

    def test_resume(self, controller):
        """Test resuming."""
        controller.set_time_scale(0.5)
        controller.pause()
        controller.resume()

        assert controller.is_paused is False
        assert controller.time_scale == 0.5  # Restored scale
        assert controller.state == TimeState.SLOW_MOTION

    def test_resume_when_not_paused(self, controller):
        """Test resuming when not paused."""
        controller.resume()  # Should not change anything
        assert controller.is_paused is False
        assert controller.time_scale == 1.0

    def test_toggle_pause(self, controller):
        """Test toggle pause."""
        result = controller.toggle_pause()
        assert result is True
        assert controller.is_paused is True

        result = controller.toggle_pause()
        assert result is False
        assert controller.is_paused is False

    def test_set_time_scale(self, controller):
        """Test setting time scale."""
        result = controller.set_time_scale(0.5)
        assert result == 0.5
        assert controller.time_scale == 0.5
        assert controller.state == TimeState.SLOW_MOTION

    def test_set_time_scale_fast_forward(self, controller):
        """Test fast forward time scale."""
        controller.set_time_scale(2.0)
        assert controller.time_scale == 2.0
        assert controller.state == TimeState.FAST_FORWARD

    def test_set_time_scale_clamped(self, controller):
        """Test time scale is clamped."""
        result = controller.set_time_scale(100.0)
        assert result == 10.0  # Max scale

        result = controller.set_time_scale(-1.0)
        assert result == 0.01  # Min scale

    def test_set_time_scale_while_paused(self, controller):
        """Test setting scale while paused."""
        controller.pause()
        controller.set_time_scale(0.5)

        # Scale is stored but time_scale remains 0
        assert controller.time_scale == 0.0
        assert controller.get_time_scale() == 0.5

        # Resume restores the stored scale
        controller.resume()
        assert controller.time_scale == 0.5

    def test_get_time_scale(self, controller):
        """Test get_time_scale returns stored scale when paused."""
        controller.set_time_scale(0.5)
        controller.pause()

        assert controller.time_scale == 0.0  # Actual scale
        assert controller.get_time_scale() == 0.5  # Stored scale

    def test_step_frame(self, controller):
        """Test frame stepping."""
        controller.pause()
        count = controller.step_frame()

        assert count == 1
        assert controller.frame_step_pending is True
        assert controller.state == TimeState.FRAME_STEP

    def test_step_frame_multiple(self, controller):
        """Test stepping multiple frames."""
        controller.pause()
        count = controller.step_frame(5)

        assert count == 5
        assert controller.frame_step_pending is True

    def test_step_frame_when_not_paused(self, controller):
        """Test step frame when not paused returns 0."""
        count = controller.step_frame()
        assert count == 0
        assert controller.frame_step_pending is False

    def test_consume_frame_step(self, controller):
        """Test consuming frame steps."""
        controller.pause()
        controller.step_frame(3)

        # Consume first step
        assert controller.consume_frame_step() is True
        assert controller.frame_step_pending is True

        # Consume second step
        assert controller.consume_frame_step() is True
        assert controller.frame_step_pending is True

        # Consume third step - should clear pending
        assert controller.consume_frame_step() is True
        assert controller.frame_step_pending is False
        assert controller.state == TimeState.PAUSED

        # No more steps
        assert controller.consume_frame_step() is False

    def test_set_slow_motion(self, controller):
        """Test slow motion preset."""
        result = controller.set_slow_motion()
        assert result == 0.25  # Default slow motion

        result = controller.set_slow_motion(0.1)
        assert result == 0.1

    def test_set_fast_forward(self, controller):
        """Test fast forward preset."""
        result = controller.set_fast_forward()
        assert result == 2.0  # Default fast forward

        result = controller.set_fast_forward(4.0)
        assert result == 4.0

    def test_reset(self, controller):
        """Test reset to normal."""
        controller.set_time_scale(0.5)
        controller.pause()
        controller.step_frame()

        controller.reset()

        assert controller.is_paused is False
        assert controller.time_scale == 1.0
        assert controller.state == TimeState.NORMAL
        assert controller.frame_step_pending is False

    def test_apply_preset(self, controller):
        """Test applying presets."""
        controller.apply_preset(controller.PRESET_SLOW)
        assert controller.time_scale == 0.25

        controller.apply_preset(controller.PRESET_NORMAL)
        assert controller.time_scale == 1.0

        controller.apply_preset(controller.PRESET_FAST)
        assert controller.time_scale == 2.0

    def test_pause_callback(self, controller):
        """Test pause callbacks."""
        callback = Mock()
        controller.add_pause_callback(callback)

        controller.pause()
        callback.assert_called_with(True)

        callback.reset_mock()
        controller.resume()
        callback.assert_called_with(False)

    def test_remove_pause_callback(self, controller):
        """Test removing pause callback."""
        callback = Mock()
        controller.add_pause_callback(callback)
        assert controller.remove_pause_callback(callback) is True
        assert controller.remove_pause_callback(callback) is False

        controller.pause()
        callback.assert_not_called()

    def test_scale_callback(self, controller):
        """Test scale callbacks."""
        callback = Mock()
        controller.add_scale_callback(callback)

        controller.set_time_scale(0.5)
        callback.assert_called_with(0.5)

    def test_state_callback(self, controller):
        """Test state callbacks."""
        callback = Mock()
        controller.add_state_callback(callback)

        controller.pause()
        callback.assert_called_with(TimeState.PAUSED)

        callback.reset_mock()
        controller.step_frame()
        callback.assert_called_with(TimeState.FRAME_STEP)

    def test_console_commands(self, controller):
        """Test console command methods."""
        result = controller.cmd_pause()
        assert "paused" in result.lower()
        assert controller.is_paused is True

        result = controller.cmd_resume()
        assert "resumed" in result.lower()
        assert controller.is_paused is False

        result = controller.cmd_slomo(0.5)
        assert "0.5" in result
        assert controller.time_scale == 0.5

        controller.pause()
        result = controller.cmd_step(2)
        assert "2" in result
        assert controller.frame_step_pending is True

    def test_cmd_step_when_not_paused(self, controller):
        """Test step command when not paused."""
        result = controller.cmd_step()
        assert "not paused" in result.lower() or "cannot" in result.lower()


class TestTimeState:
    """Tests for TimeState enum."""

    def test_states(self):
        """Test all time states exist."""
        assert TimeState.NORMAL
        assert TimeState.PAUSED
        assert TimeState.SLOW_MOTION
        assert TimeState.FAST_FORWARD
        assert TimeState.FRAME_STEP


class TestGetTimeController:
    """Tests for get_time_controller singleton."""

    def test_singleton(self):
        """Test that get_time_controller returns singleton."""
        # Reset the singleton
        import engine.debug.tools.time_control as time_module
        time_module._time_controller = None

        controller1 = get_time_controller()
        controller2 = get_time_controller()
        assert controller1 is controller2


class TestBuildTypeGuards:
    """Tests for build-type security guards in TimeController."""

    @pytest.fixture(autouse=True)
    def reset_env(self):
        """Reset environment before each test."""
        import os
        for var in ["GAME_BUILD_TYPE", "SHIPPING"]:
            os.environ.pop(var, None)
        yield
        for var in ["GAME_BUILD_TYPE", "SHIPPING"]:
            os.environ.pop(var, None)

    def test_pause_blocked_in_shipping(self):
        """Test pause is blocked in shipping builds."""
        import os
        os.environ["GAME_BUILD_TYPE"] = "SHIPPING"

        controller = TimeController()
        controller.pause()

        # Pause should be blocked
        assert controller.is_paused is False

    def test_time_scale_blocked_in_shipping(self):
        """Test time scale changes are blocked in shipping builds."""
        import os
        os.environ["GAME_BUILD_TYPE"] = "SHIPPING"

        controller = TimeController()
        result = controller.set_time_scale(0.5)

        # Should return normal scale, not requested scale
        assert result == 1.0
        assert controller.time_scale == 1.0

    def test_allowed_in_development(self):
        """Test time control works in development builds."""
        import os
        os.environ["GAME_BUILD_TYPE"] = "DEVELOPMENT"

        controller = TimeController()
        controller.pause()

        assert controller.is_paused is True

    def test_config_allows_shipping_override(self):
        """Test config can allow time control in shipping."""
        import os
        os.environ["GAME_BUILD_TYPE"] = "SHIPPING"

        config = TimeControlConfig(allow_in_shipping=True)
        controller = TimeController(config)

        controller.pause()
        assert controller.is_paused is True


class TestTimeControlPresets:
    """Tests for time control presets via config."""

    def test_presets_from_config(self):
        """Test that presets come from config, not hardcoded."""
        config = TimeControlConfig(
            preset_slow=0.1,
            preset_fast=5.0,
        )
        controller = TimeController(config)

        assert controller.PRESET_SLOW == 0.1
        assert controller.PRESET_FAST == 5.0

    def test_default_presets(self):
        """Test default preset values."""
        controller = TimeController()

        assert controller.PRESET_SUPER_SLOW == 0.1
        assert controller.PRESET_SLOW == 0.25
        assert controller.PRESET_HALF == 0.5
        assert controller.PRESET_NORMAL == 1.0
        assert controller.PRESET_FAST == 2.0
        assert controller.PRESET_SUPER_FAST == 4.0


class TestTimeControlGameImpact:
    """
    Tests that verify time control actually impacts game timing,
    not just internal state.
    """

    @pytest.fixture
    def controller(self):
        """Create a fresh TimeController."""
        return TimeController()

    def test_pause_affects_game_loop(self, controller):
        """Test pause actually affects a mock game loop."""
        game_ticks = 0

        def game_tick():
            nonlocal game_ticks
            if not controller.is_paused:
                game_ticks += 1

        # Game runs normally
        for _ in range(5):
            game_tick()
        assert game_ticks == 5

        # Pause stops game ticks
        controller.pause()
        for _ in range(5):
            game_tick()
        assert game_ticks == 5  # No additional ticks

        # Resume allows ticks again
        controller.resume()
        for _ in range(5):
            game_tick()
        assert game_ticks == 10

    def test_time_scale_affects_delta_time(self, controller):
        """Test time scale affects delta time calculations."""
        real_dt = 0.016  # ~60fps

        def get_scaled_dt():
            return real_dt * controller.time_scale

        # Normal scale
        assert get_scaled_dt() == real_dt

        # Slow motion
        controller.set_time_scale(0.5)
        assert get_scaled_dt() == real_dt * 0.5

        # Fast forward
        controller.set_time_scale(2.0)
        assert get_scaled_dt() == real_dt * 2.0

    def test_frame_step_advances_single_frame(self, controller):
        """Test frame stepping advances exactly one frame."""
        frames_processed = 0

        def process_frame():
            nonlocal frames_processed
            if controller.consume_frame_step() or not controller.is_paused:
                frames_processed += 1

        # Pause and step
        controller.pause()
        controller.step_frame(1)

        # Process frames
        for _ in range(5):
            process_frame()

        # Only 1 frame should have processed
        assert frames_processed == 1
