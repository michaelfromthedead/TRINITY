"""Tests for XR session state machine management."""
from __future__ import annotations

import pytest
import sys

sys.path.insert(0, "/home/user/dev/AI_GAME_ENGINE")

from engine.xr.runtime.session import (
    XRSessionState,
    XRSessionMode,
    XRReferenceSpace,
    XRSessionConfig,
    XRSessionStats,
    XRSession,
    XRSessionError,
    InvalidStateTransitionError,
)
from engine.xr.runtime.capabilities import (
    XRCapabilities,
    XRFeature,
)


@pytest.fixture
def basic_capabilities() -> XRCapabilities:
    """Create basic XR capabilities for testing."""
    return XRCapabilities(
        device_name="Test Device",
        features=frozenset({
            XRFeature.HEAD_TRACKING,
            XRFeature.CONTROLLER_TRACKING,
            XRFeature.HAND_TRACKING,
        }),
    )


@pytest.fixture
def full_capabilities() -> XRCapabilities:
    """Create full XR capabilities for testing."""
    return XRCapabilities(
        device_name="Full Test Device",
        features=frozenset({
            XRFeature.HEAD_TRACKING,
            XRFeature.CONTROLLER_TRACKING,
            XRFeature.HAND_TRACKING,
            XRFeature.EYE_TRACKING,
            XRFeature.FOVEATED_RENDERING,
            XRFeature.DYNAMIC_FOVEATION,
            XRFeature.PASSTHROUGH,
        }),
    )


class TestXRSessionState:
    """Tests for XRSessionState enumeration."""

    def test_all_states_defined(self):
        """Verify all expected states exist."""
        assert XRSessionState.IDLE
        assert XRSessionState.READY
        assert XRSessionState.RUNNING
        assert XRSessionState.PAUSED
        assert XRSessionState.STOPPING
        assert XRSessionState.ERROR


class TestXRSessionMode:
    """Tests for XRSessionMode enumeration."""

    def test_all_modes_defined(self):
        """Verify all session modes exist."""
        assert XRSessionMode.IMMERSIVE_VR
        assert XRSessionMode.IMMERSIVE_AR
        assert XRSessionMode.INLINE


class TestXRReferenceSpace:
    """Tests for XRReferenceSpace enumeration."""

    def test_all_spaces_defined(self):
        """Verify all reference spaces exist."""
        assert XRReferenceSpace.LOCAL
        assert XRReferenceSpace.LOCAL_FLOOR
        assert XRReferenceSpace.BOUNDED_FLOOR
        assert XRReferenceSpace.UNBOUNDED
        assert XRReferenceSpace.VIEWER


class TestXRSessionConfig:
    """Tests for XRSessionConfig dataclass."""

    def test_default_values(self):
        """Verify default configuration values."""
        config = XRSessionConfig()
        assert config.mode == XRSessionMode.IMMERSIVE_VR
        assert config.reference_space == XRReferenceSpace.LOCAL_FLOOR
        assert config.enable_hand_tracking is True
        assert config.enable_eye_tracking is False
        assert config.render_scale == 1.0
        assert config.target_refresh_rate == 90.0

    def test_custom_values(self):
        """Verify custom configuration values."""
        config = XRSessionConfig(
            mode=XRSessionMode.IMMERSIVE_AR,
            enable_passthrough=True,
            render_scale=1.5,
            foveation_level=3,
        )
        assert config.mode == XRSessionMode.IMMERSIVE_AR
        assert config.enable_passthrough is True
        assert config.render_scale == 1.5
        assert config.foveation_level == 3


class TestXRSessionStats:
    """Tests for XRSessionStats dataclass."""

    def test_default_values(self):
        """Verify default stats values."""
        stats = XRSessionStats()
        assert stats.frames_presented == 0
        assert stats.dropped_frames == 0
        assert stats.start_time == 0.0
        assert stats.average_framerate == 0.0

    def test_mutable_stats(self):
        """Verify stats can be updated."""
        stats = XRSessionStats()
        stats.frames_presented = 100
        stats.dropped_frames = 5
        assert stats.frames_presented == 100
        assert stats.dropped_frames == 5


class TestXRSessionLifecycle:
    """Tests for XRSession lifecycle management."""

    def test_initial_state_is_idle(self):
        """Verify session starts in IDLE state."""
        config = XRSessionConfig()
        session = XRSession(config)
        assert session.state == XRSessionState.IDLE

    def test_initialize_transitions_to_ready(self, basic_capabilities):
        """Verify initialize() transitions to READY."""
        session = XRSession(XRSessionConfig())
        assert session.initialize(basic_capabilities) is True
        assert session.state == XRSessionState.READY

    def test_start_transitions_to_running(self, basic_capabilities):
        """Verify start() transitions to RUNNING."""
        session = XRSession(XRSessionConfig())
        session.initialize(basic_capabilities)
        assert session.start() is True
        assert session.state == XRSessionState.RUNNING
        assert session.is_running is True

    def test_pause_transitions_to_paused(self, basic_capabilities):
        """Verify pause() transitions to PAUSED."""
        session = XRSession(XRSessionConfig())
        session.initialize(basic_capabilities)
        session.start()
        assert session.pause() is True
        assert session.state == XRSessionState.PAUSED
        assert session.is_paused is True

    def test_resume_transitions_to_running(self, basic_capabilities):
        """Verify resume() transitions back to RUNNING."""
        session = XRSession(XRSessionConfig())
        session.initialize(basic_capabilities)
        session.start()
        session.pause()
        assert session.resume() is True
        assert session.state == XRSessionState.RUNNING

    def test_stop_transitions_to_idle(self, basic_capabilities):
        """Verify stop() transitions through STOPPING to IDLE."""
        session = XRSession(XRSessionConfig())
        session.initialize(basic_capabilities)
        session.start()
        assert session.stop() is True
        assert session.state == XRSessionState.IDLE

    def test_stop_from_paused(self, basic_capabilities):
        """Verify stop() works from PAUSED state."""
        session = XRSession(XRSessionConfig())
        session.initialize(basic_capabilities)
        session.start()
        session.pause()
        assert session.stop() is True
        assert session.state == XRSessionState.IDLE


class TestXRSessionStateValidation:
    """Tests for state transition validation."""

    def test_cannot_start_without_initialize(self):
        """Verify start() fails if not initialized."""
        session = XRSession(XRSessionConfig())
        assert session.start() is False
        assert session.state == XRSessionState.IDLE

    def test_cannot_pause_when_not_running(self, basic_capabilities):
        """Verify pause() fails if not running."""
        session = XRSession(XRSessionConfig())
        session.initialize(basic_capabilities)
        assert session.pause() is False
        assert session.state == XRSessionState.READY

    def test_cannot_resume_when_not_paused(self, basic_capabilities):
        """Verify resume() fails if not paused."""
        session = XRSession(XRSessionConfig())
        session.initialize(basic_capabilities)
        session.start()
        assert session.resume() is False
        assert session.state == XRSessionState.RUNNING

    def test_can_transition_to_check(self, basic_capabilities):
        """Verify can_transition_to() correctly validates transitions."""
        session = XRSession(XRSessionConfig())

        # From IDLE
        assert session.can_transition_to(XRSessionState.READY) is True
        assert session.can_transition_to(XRSessionState.RUNNING) is False

        session.initialize(basic_capabilities)

        # From READY
        assert session.can_transition_to(XRSessionState.RUNNING) is True
        assert session.can_transition_to(XRSessionState.IDLE) is False


class TestXRSessionHooks:
    """Tests for state enter/exit hooks."""

    def test_enter_hook_called_on_transition(self, basic_capabilities):
        """Verify enter hooks are called on state transition."""
        session = XRSession(XRSessionConfig())
        entered_states = []

        def on_enter(s):
            entered_states.append(s.state)

        session.add_enter_hook(XRSessionState.READY, on_enter)
        session.add_enter_hook(XRSessionState.RUNNING, on_enter)

        session.initialize(basic_capabilities)
        assert XRSessionState.READY in entered_states

        session.start()
        assert XRSessionState.RUNNING in entered_states

    def test_exit_hook_called_on_transition(self, basic_capabilities):
        """Verify exit hooks are called on state transition."""
        session = XRSession(XRSessionConfig())
        exited_states = []

        def on_exit(s):
            # s.state is already the new state when exit hook is called
            exited_states.append(True)

        session.add_exit_hook(XRSessionState.READY, on_exit)

        session.initialize(basic_capabilities)
        session.start()

        assert len(exited_states) == 1

    def test_remove_enter_hook(self, basic_capabilities):
        """Verify enter hooks can be removed."""
        session = XRSession(XRSessionConfig())
        calls = []

        def on_enter(s):
            calls.append(1)

        session.add_enter_hook(XRSessionState.READY, on_enter)
        assert session.remove_enter_hook(XRSessionState.READY, on_enter) is True

        session.initialize(basic_capabilities)
        assert len(calls) == 0

    def test_remove_nonexistent_hook_returns_false(self):
        """Verify removing nonexistent hook returns False."""
        session = XRSession(XRSessionConfig())

        def dummy(s):
            pass

        assert session.remove_enter_hook(XRSessionState.READY, dummy) is False
        assert session.remove_exit_hook(XRSessionState.READY, dummy) is False


class TestXRSessionFeatures:
    """Tests for feature enablement based on capabilities."""

    def test_features_enabled_based_on_config_and_caps(self, full_capabilities):
        """Verify features are enabled based on config and capabilities."""
        config = XRSessionConfig(
            enable_hand_tracking=True,
            enable_eye_tracking=True,
            enable_foveation=True,
        )
        session = XRSession(config)
        session.initialize(full_capabilities)

        assert session.is_feature_enabled(XRFeature.HEAD_TRACKING)
        assert session.is_feature_enabled(XRFeature.HAND_TRACKING)
        assert session.is_feature_enabled(XRFeature.EYE_TRACKING)
        assert session.is_feature_enabled(XRFeature.FOVEATED_RENDERING)

    def test_feature_not_enabled_if_not_in_config(self, full_capabilities):
        """Verify features not enabled if config disables them."""
        config = XRSessionConfig(
            enable_hand_tracking=False,
            enable_eye_tracking=False,
        )
        session = XRSession(config)
        session.initialize(full_capabilities)

        assert not session.is_feature_enabled(XRFeature.HAND_TRACKING)
        assert not session.is_feature_enabled(XRFeature.EYE_TRACKING)

    def test_feature_not_enabled_if_not_supported(self, basic_capabilities):
        """Verify features not enabled if not supported by device."""
        config = XRSessionConfig(
            enable_eye_tracking=True,  # Request it
        )
        session = XRSession(config)
        session.initialize(basic_capabilities)  # But device doesn't support it

        assert not session.is_feature_enabled(XRFeature.EYE_TRACKING)


class TestXRSessionStats:
    """Tests for session statistics tracking."""

    def test_update_frame_stats_presented(self, basic_capabilities):
        """Verify frame stats update for presented frames."""
        session = XRSession(XRSessionConfig())
        session.initialize(basic_capabilities)
        session.start()

        session.update_frame_stats(frame_presented=True)
        session.update_frame_stats(frame_presented=True)

        assert session.stats.frames_presented == 2
        assert session.stats.dropped_frames == 0

    def test_update_frame_stats_dropped(self, basic_capabilities):
        """Verify frame stats update for dropped frames."""
        session = XRSession(XRSessionConfig())
        session.initialize(basic_capabilities)
        session.start()

        session.update_frame_stats(frame_presented=False)
        session.update_frame_stats(frame_presented=True)

        assert session.stats.frames_presented == 1
        assert session.stats.dropped_frames == 1

    def test_update_frame_stats_reprojected(self, basic_capabilities):
        """Verify frame stats track reprojected frames."""
        session = XRSession(XRSessionConfig())
        session.initialize(basic_capabilities)
        session.start()

        session.update_frame_stats(frame_presented=True, was_reprojected=True)
        session.update_frame_stats(frame_presented=True, was_reprojected=False)

        assert session.stats.reprojection_count == 1


class TestXRSessionStateHistory:
    """Tests for state history tracking."""

    def test_state_history_recorded(self, basic_capabilities):
        """Verify state transitions are recorded in history."""
        session = XRSession(XRSessionConfig())
        session.initialize(basic_capabilities)
        session.start()
        session.pause()
        session.resume()
        session.stop()

        history = session.get_state_history()

        # Should have: READY, RUNNING, PAUSED, RUNNING, STOPPING, IDLE
        states = [state for state, _ in history]
        assert XRSessionState.READY in states
        assert XRSessionState.RUNNING in states
        assert XRSessionState.PAUSED in states

    def test_state_history_has_timestamps(self, basic_capabilities):
        """Verify state history entries have timestamps."""
        session = XRSession(XRSessionConfig())
        session.initialize(basic_capabilities)

        history = session.get_state_history()
        assert len(history) > 0

        state, timestamp = history[0]
        assert isinstance(timestamp, float)
        assert timestamp > 0


class TestXRSessionErrorHandling:
    """Tests for error state handling."""

    def test_reset_from_error_state(self, basic_capabilities):
        """Verify reset() works from ERROR state."""
        session = XRSession(XRSessionConfig())
        session.initialize(basic_capabilities)
        session.start()

        # Force error state (normally this would happen internally)
        session._set_error("Test error")
        assert session.state == XRSessionState.ERROR
        assert session.error_message == "Test error"

        assert session.reset() is True
        assert session.state == XRSessionState.IDLE
        assert session.error_message is None

    def test_cannot_reset_from_non_error_state(self, basic_capabilities):
        """Verify reset() fails from non-ERROR states."""
        session = XRSession(XRSessionConfig())
        session.initialize(basic_capabilities)

        assert session.reset() is False


class TestInvalidStateTransitionError:
    """Tests for InvalidStateTransitionError exception."""

    def test_error_contains_states(self):
        """Verify error contains from/to state information."""
        error = InvalidStateTransitionError(
            XRSessionState.IDLE,
            XRSessionState.RUNNING,
        )
        assert error.from_state == XRSessionState.IDLE
        assert error.to_state == XRSessionState.RUNNING
        assert "IDLE" in str(error)
        assert "RUNNING" in str(error)
