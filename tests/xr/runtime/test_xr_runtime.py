"""Tests for XR runtime abstraction layer."""
from __future__ import annotations

import pytest
import sys

sys.path.insert(0, "/home/user/dev/AI_GAME_ENGINE")

from engine.xr.runtime import (
    XRRuntimeType,
    XRRuntimeState,
    XRRuntime,
    XRRuntimeError,
    XRRuntimeNotAvailableError,
    Pose,
    ViewInfo,
    create_runtime,
    detect_available_runtimes,
    XRSessionConfig,
    XRSessionMode,
    XRFeature,
    OpenXRRuntime,
    WebXRRuntime,
)


class TestPose:
    """Tests for Pose dataclass."""

    def test_identity_pose(self):
        """Verify identity pose has expected values."""
        pose = Pose.identity()
        assert pose.position == (0.0, 0.0, 0.0)
        assert pose.orientation == (0.0, 0.0, 0.0, 1.0)
        assert pose.is_valid is True

    def test_invalid_pose(self):
        """Verify invalid pose marker."""
        pose = Pose.invalid()
        assert pose.is_valid is False

    def test_custom_pose(self):
        """Verify custom pose values."""
        pose = Pose(
            position=(1.0, 2.0, 3.0),
            orientation=(0.0, 0.707, 0.0, 0.707),
            linear_velocity=(0.1, 0.0, 0.0),
        )
        assert pose.position == (1.0, 2.0, 3.0)
        assert pose.orientation[1] == 0.707

    def test_pose_is_immutable(self):
        """Verify pose is frozen/immutable."""
        pose = Pose.identity()
        with pytest.raises(AttributeError):
            pose.position = (1.0, 0.0, 0.0)


class TestViewInfo:
    """Tests for ViewInfo dataclass."""

    def test_default_view_info(self):
        """Verify default view info values."""
        view = ViewInfo()
        assert view.near_clip == 0.1
        assert view.far_clip == 1000.0
        assert len(view.fov) == 4

    def test_custom_view_info(self):
        """Verify custom view info values."""
        pose = Pose(position=(0.032, 1.6, 0.0))
        view = ViewInfo(
            pose=pose,
            near_clip=0.01,
            far_clip=100.0,
        )
        assert view.pose.position[0] == 0.032
        assert view.near_clip == 0.01


class TestXRRuntimeState:
    """Tests for XRRuntimeState resource."""

    def test_default_state(self):
        """Verify default runtime state values."""
        state = XRRuntimeState()
        assert state.runtime_name == ""
        assert state.session_state == "idle"
        assert state.display_refresh_rate == 90.0

    def test_update_from_session(self):
        """Verify state updates from session."""
        from engine.xr.runtime.session import XRSession
        from engine.xr.runtime.capabilities import XRCapabilities, DisplaySpecs

        caps = XRCapabilities(
            features=frozenset({XRFeature.HEAD_TRACKING, XRFeature.HAND_TRACKING}),
            display=DisplaySpecs(refresh_rate=120.0),
        )
        config = XRSessionConfig(render_scale=1.5)
        session = XRSession(config)
        session.initialize(caps)

        state = XRRuntimeState()
        state.update_from_session(session)

        assert state.session_state == "ready"
        assert state.display_refresh_rate == 120.0
        assert state.render_scale == 1.5
        assert state.supports_hand_tracking is True


class TestOpenXRRuntime:
    """Tests for OpenXR runtime implementation."""

    def test_initialization(self):
        """Verify OpenXR runtime initializes."""
        runtime = OpenXRRuntime()
        assert runtime.initialize() is True
        assert runtime.is_available is True
        assert runtime.runtime_type == XRRuntimeType.OPENXR
        runtime.shutdown()

    def test_capabilities_populated(self):
        """Verify capabilities are populated after init."""
        runtime = OpenXRRuntime()
        runtime.initialize()
        assert runtime.capabilities is not None
        assert runtime.capabilities.supports(XRFeature.HEAD_TRACKING)
        runtime.shutdown()

    def test_session_creation(self):
        """Verify session can be created."""
        runtime = OpenXRRuntime()
        runtime.initialize()
        session = runtime.create_session()
        assert session is not None
        assert runtime.session is session
        runtime.shutdown()

    def test_session_lifecycle(self):
        """Verify full session lifecycle."""
        runtime = OpenXRRuntime()
        runtime.initialize()
        runtime.create_session()

        assert runtime.start_session() is True
        assert runtime.is_session_active is True

        assert runtime.stop_session() is True
        assert runtime.is_session_active is False

        runtime.shutdown()

    def test_frame_cycle(self):
        """Verify frame wait/begin/end cycle."""
        runtime = OpenXRRuntime()
        runtime.initialize()
        runtime.create_session()
        runtime.start_session()

        assert runtime.wait_frame() is True
        assert runtime.begin_frame() is True
        assert runtime.end_frame() is True

        runtime.shutdown()

    def test_tracking_methods(self):
        """Verify tracking methods return valid data."""
        runtime = OpenXRRuntime()
        runtime.initialize()
        runtime.create_session()
        runtime.start_session()

        head_pose = runtime.get_head_pose()
        assert head_pose.is_valid is True
        assert head_pose.position[1] > 0  # Should be at some height

        left_view = runtime.get_view_info(0)
        right_view = runtime.get_view_info(1)
        # Left and right eyes should have different X positions
        assert left_view.pose.position[0] != right_view.pose.position[0]

        runtime.shutdown()

    def test_destroy_session(self):
        """Verify session destruction."""
        runtime = OpenXRRuntime()
        runtime.initialize()
        runtime.create_session()
        runtime.start_session()

        runtime.destroy_session()
        assert runtime.session is None

        runtime.shutdown()


class TestWebXRRuntime:
    """Tests for WebXR runtime implementation."""

    def test_initialization(self):
        """Verify WebXR runtime initializes."""
        runtime = WebXRRuntime()
        assert runtime.initialize() is True
        assert runtime.is_available is True
        assert runtime.runtime_type == XRRuntimeType.WEBXR
        runtime.shutdown()

    def test_session_mode_request(self):
        """Verify session mode can be requested."""
        from engine.xr.runtime.webxr import WebXRSessionMode

        runtime = WebXRRuntime()
        runtime.initialize()

        assert runtime.request_session(WebXRSessionMode.IMMERSIVE_VR) is True
        assert runtime.session_mode == WebXRSessionMode.IMMERSIVE_VR

        runtime.shutdown()

    def test_input_sources(self):
        """Verify input sources are available."""
        runtime = WebXRRuntime()
        runtime.initialize()
        runtime.create_session()
        runtime.start_session()
        runtime.begin_frame()

        left = runtime.get_input_source("left")
        right = runtime.get_input_source("right")

        assert left is not None
        assert right is not None
        assert left.handedness == "left"
        assert right.handedness == "right"

        runtime.shutdown()


class TestCreateRuntime:
    """Tests for create_runtime() factory function."""

    def test_create_openxr_runtime(self):
        """Verify OpenXR runtime creation."""
        runtime = create_runtime(XRRuntimeType.OPENXR)
        assert runtime.runtime_type == XRRuntimeType.OPENXR
        runtime.shutdown()

    def test_create_webxr_runtime(self):
        """Verify WebXR runtime creation."""
        runtime = create_runtime(XRRuntimeType.WEBXR)
        assert runtime.runtime_type == XRRuntimeType.WEBXR
        runtime.shutdown()

    def test_create_mock_runtime(self):
        """Verify mock runtime creation."""
        runtime = create_runtime(XRRuntimeType.MOCK)
        assert runtime.runtime_type == XRRuntimeType.MOCK
        runtime.shutdown()

    def test_fallback_to_mock(self):
        """Verify fallback to mock when primary unavailable."""
        # This test assumes we can create any runtime in simulation mode
        runtime = create_runtime(XRRuntimeType.OPENXR, fallback=True)
        assert runtime is not None
        runtime.shutdown()


class TestDetectAvailableRuntimes:
    """Tests for detect_available_runtimes() function."""

    def test_mock_always_available(self):
        """Verify mock runtime is always in available list."""
        runtimes = detect_available_runtimes()
        assert XRRuntimeType.MOCK in runtimes

    def test_returns_list(self):
        """Verify function returns a list."""
        runtimes = detect_available_runtimes()
        assert isinstance(runtimes, list)


class TestRuntimeEventSystem:
    """Tests for runtime event emission."""

    def test_event_registration(self):
        """Verify events can be registered."""
        runtime = create_runtime(XRRuntimeType.MOCK)
        runtime.initialize()

        events_received = []

        def handler(event_data):
            events_received.append(event_data)

        runtime.on("test_event", handler)
        runtime.emit("test_event", "data")

        assert len(events_received) == 1
        assert events_received[0] == "data"

        runtime.shutdown()

    def test_event_unregistration(self):
        """Verify events can be unregistered."""
        runtime = create_runtime(XRRuntimeType.MOCK)
        runtime.initialize()

        events_received = []

        def handler(event_data):
            events_received.append(event_data)

        runtime.on("test_event", handler)
        assert runtime.off("test_event", handler) is True
        runtime.emit("test_event", "data")

        assert len(events_received) == 0

        runtime.shutdown()


class TestRuntimeFeatureQueries:
    """Tests for runtime feature query methods."""

    def test_supports_feature(self):
        """Verify supports_feature() method."""
        runtime = create_runtime(XRRuntimeType.OPENXR)
        runtime.initialize()

        assert runtime.supports_feature(XRFeature.HEAD_TRACKING) is True

        runtime.shutdown()

    def test_is_feature_enabled(self):
        """Verify is_feature_enabled() method."""
        runtime = create_runtime(XRRuntimeType.OPENXR)
        runtime.initialize()

        config = XRSessionConfig(enable_hand_tracking=True)
        runtime.create_session(config)

        if runtime.supports_feature(XRFeature.HAND_TRACKING):
            assert runtime.is_feature_enabled(XRFeature.HAND_TRACKING) is True

        runtime.shutdown()


class TestRuntimeSettings:
    """Tests for runtime settings methods."""

    def test_set_render_scale(self):
        """Verify render scale can be set."""
        runtime = create_runtime(XRRuntimeType.MOCK)
        runtime.initialize()
        runtime.create_session()

        runtime.set_render_scale(1.5)
        assert runtime.state.render_scale == 1.5

        # Test clamping
        runtime.set_render_scale(3.0)  # Should clamp to 2.0
        assert runtime.state.render_scale == 2.0

        runtime.set_render_scale(0.1)  # Should clamp to 0.5
        assert runtime.state.render_scale == 0.5

        runtime.shutdown()

    def test_set_refresh_rate(self):
        """Verify refresh rate can be set."""
        runtime = create_runtime(XRRuntimeType.MOCK)
        runtime.initialize()
        runtime.create_session()

        if runtime.capabilities:
            valid_rate = runtime.capabilities.display.supported_refresh_rates[0]
            assert runtime.set_refresh_rate(valid_rate) is True

            # Invalid rate should fail
            assert runtime.set_refresh_rate(999.0) is False

        runtime.shutdown()


class TestRuntimeBoundary:
    """Tests for boundary/guardian methods."""

    def test_boundary_geometry(self):
        """Verify boundary geometry can be queried."""
        runtime = create_runtime(XRRuntimeType.OPENXR)
        runtime.initialize()
        runtime.create_session()
        runtime.start_session()

        boundary = runtime.get_boundary_geometry()
        if boundary is not None:
            assert len(boundary) >= 3  # At least a triangle
            # Each point should be a 3-tuple
            for point in boundary:
                assert len(point) == 3

        runtime.shutdown()


class TestRuntimeErrors:
    """Tests for runtime error handling."""

    def test_create_session_without_init_raises(self):
        """Verify session creation without init raises error."""
        runtime = OpenXRRuntime()
        # Don't initialize

        with pytest.raises(XRRuntimeError):
            runtime.create_session()

    def test_double_session_creation_raises(self):
        """Verify creating second session raises error."""
        runtime = create_runtime(XRRuntimeType.MOCK)
        runtime.initialize()
        runtime.create_session()

        with pytest.raises(XRRuntimeError):
            runtime.create_session()

        runtime.shutdown()
