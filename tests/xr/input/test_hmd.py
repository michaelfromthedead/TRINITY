"""
Tests for HeadMountedDisplay component.

Tests HMD pose tracking, prediction, view matrix computation,
and tracking state machine.
"""

import pytest
import math

from engine.xr.input.hmd import (
    HeadMountedDisplay,
    HMDTrackingState,
    HMD_STATE_TRANSITIONS,
    HMDDisplayInfo,
    PredictionConfig,
)
from trinity.descriptors.tracking import is_dirty, clear_dirty, get_dirty_fields


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestHMDInitialization:
    """Test HMD component initialization."""

    def test_default_initialization(self):
        """Test HMD initializes with default values."""
        hmd = HeadMountedDisplay()

        assert hmd.position == (0.0, 0.0, 0.0)
        assert hmd.orientation == (0.0, 0.0, 0.0, 1.0)
        assert hmd.linear_velocity == (0.0, 0.0, 0.0)
        assert hmd.angular_velocity == (0.0, 0.0, 0.0)
        assert hmd.tracking_state == HMDTrackingState.INITIALIZING
        assert hmd.confidence == 0.0

    def test_device_id(self):
        """Test device ID is stored correctly."""
        hmd = HeadMountedDisplay(device_id="test_hmd_001")
        assert hmd.device_id == "test_hmd_001"

    def test_custom_display_info(self):
        """Test custom display info is applied."""
        display_info = HMDDisplayInfo(
            resolution_per_eye=(2160, 2160),
            refresh_rate=120.0,
            field_of_view=(110.0, 100.0),
            ipd=0.065,
        )
        hmd = HeadMountedDisplay(display_info=display_info)

        assert hmd.display_info.resolution_per_eye == (2160, 2160)
        assert hmd.display_info.refresh_rate == 120.0
        assert hmd.display_info.field_of_view == (110.0, 100.0)
        assert hmd.display_info.ipd == 0.065

    def test_custom_prediction_config(self):
        """Test custom prediction config is applied."""
        config = PredictionConfig(
            enabled=True,
            prediction_time_ms=8.0,
            max_prediction_time_ms=25.0,
        )
        hmd = HeadMountedDisplay(prediction_config=config)

        assert hmd.prediction_config.enabled is True
        assert hmd.prediction_config.prediction_time_ms == 8.0
        assert hmd.prediction_config.max_prediction_time_ms == 25.0


# =============================================================================
# POSE UPDATE TESTS
# =============================================================================


class TestHMDPoseUpdate:
    """Test HMD pose updates."""

    def test_update_position(self):
        """Test position update."""
        hmd = HeadMountedDisplay()
        hmd.update_pose(
            position=(1.0, 1.5, -2.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            confidence=1.0,
        )

        assert hmd.position == (1.0, 1.5, -2.0)

    def test_update_orientation(self):
        """Test orientation update."""
        hmd = HeadMountedDisplay()
        # 90 degree rotation around Y axis
        q = (0.0, 0.7071, 0.0, 0.7071)
        hmd.update_pose(
            position=(0.0, 0.0, 0.0),
            orientation=q,
            confidence=1.0,
        )

        assert hmd.orientation == q

    def test_update_velocity(self):
        """Test velocity is provided or calculated."""
        hmd = HeadMountedDisplay()
        hmd.update_pose(
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            linear_velocity=(1.0, 0.0, -1.0),
            angular_velocity=(0.0, 0.5, 0.0),
            confidence=1.0,
        )

        assert hmd.linear_velocity == (1.0, 0.0, -1.0)
        assert hmd.angular_velocity == (0.0, 0.5, 0.0)

    def test_velocity_calculation(self):
        """Test velocity is calculated from position delta."""
        hmd = HeadMountedDisplay()

        # First update at t=0
        hmd.update_pose(
            position=(0.0, 1.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            confidence=1.0,
            timestamp=0.0,
        )

        # Second update at t=0.1
        hmd.update_pose(
            position=(1.0, 1.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            confidence=1.0,
            timestamp=0.1,
        )

        # Velocity should be ~(10, 0, 0) m/s
        assert abs(hmd.linear_velocity[0] - 10.0) < 0.01
        assert abs(hmd.linear_velocity[1]) < 0.01
        assert abs(hmd.linear_velocity[2]) < 0.01

    def test_confidence_clamping(self):
        """Test confidence is clamped to 0-1."""
        hmd = HeadMountedDisplay()

        hmd.update_pose(
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            confidence=1.5,
        )
        assert hmd.confidence == 1.0

        hmd.update_pose(
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            confidence=-0.5,
        )
        assert hmd.confidence == 0.0


# =============================================================================
# TRACKING STATE TESTS
# =============================================================================


class TestHMDTrackingState:
    """Test HMD tracking state machine."""

    def test_initial_state(self):
        """Test initial state is INITIALIZING."""
        hmd = HeadMountedDisplay()
        assert hmd.tracking_state == HMDTrackingState.INITIALIZING

    def test_transition_to_tracking(self):
        """Test transition to TRACKING on high confidence."""
        hmd = HeadMountedDisplay()
        hmd.update_pose(
            position=(0.0, 1.5, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            confidence=0.95,
        )
        assert hmd.tracking_state == HMDTrackingState.TRACKING

    def test_transition_to_limited(self):
        """Test transition to LIMITED on medium confidence."""
        hmd = HeadMountedDisplay()

        # First get to tracking
        hmd.update_pose(
            position=(0.0, 1.5, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            confidence=0.95,
        )

        # Then degrade to limited
        hmd.update_pose(
            position=(0.0, 1.5, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            confidence=0.6,
        )
        assert hmd.tracking_state == HMDTrackingState.LIMITED

    def test_transition_to_lost(self):
        """Test transition to LOST on zero confidence."""
        hmd = HeadMountedDisplay()
        hmd.update_pose(
            position=(0.0, 1.5, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            confidence=0.0,
        )
        assert hmd.tracking_state == HMDTrackingState.LOST

    def test_is_tracking_property(self):
        """Test is_tracking property."""
        hmd = HeadMountedDisplay()
        assert hmd.is_tracking is False

        hmd.update_pose(
            position=(0.0, 1.5, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            confidence=0.95,
        )
        assert hmd.is_tracking is True

    def test_tracking_state_callback(self):
        """Test tracking state change callback."""
        hmd = HeadMountedDisplay()
        state_changes = []

        hmd.on_tracking_state_changed(
            lambda old, new: state_changes.append((old, new))
        )

        hmd.update_pose(
            position=(0.0, 1.5, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            confidence=0.95,
        )

        assert len(state_changes) == 1
        assert state_changes[0] == (HMDTrackingState.INITIALIZING, HMDTrackingState.TRACKING)

    def test_tracking_lost_callback(self):
        """Test tracking lost callback."""
        hmd = HeadMountedDisplay()
        lost_events = []

        hmd.on_tracking_lost(lambda h: lost_events.append(h))

        hmd.update_pose(
            position=(0.0, 1.5, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            confidence=0.0,
        )

        assert len(lost_events) == 1

    def test_enable_disable(self):
        """Test enable/disable tracking."""
        hmd = HeadMountedDisplay()

        hmd.update_pose(
            position=(0.0, 1.5, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            confidence=0.95,
        )
        assert hmd.tracking_state == HMDTrackingState.TRACKING

        hmd.disable()
        assert hmd.tracking_state == HMDTrackingState.DISABLED

        hmd.enable()
        assert hmd.tracking_state == HMDTrackingState.INITIALIZING


# =============================================================================
# PREDICTION TESTS
# =============================================================================


class TestHMDPrediction:
    """Test HMD pose prediction."""

    def test_prediction_enabled(self):
        """Test prediction is enabled by default."""
        hmd = HeadMountedDisplay()
        assert hmd.prediction_config.enabled is True

    def test_predicted_position(self):
        """Test predicted position with velocity."""
        hmd = HeadMountedDisplay(
            prediction_config=PredictionConfig(
                enabled=True,
                prediction_time_ms=10.0,  # 10ms prediction
            )
        )

        hmd.update_pose(
            position=(0.0, 1.5, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            linear_velocity=(1.0, 0.0, 0.0),  # 1 m/s forward
            confidence=1.0,
        )

        # Expected: 0.0 + 1.0 * 0.01 = 0.01 meters in X
        predicted = hmd.predicted_position
        assert abs(predicted[0] - 0.01) < 0.001

    def test_prediction_disabled(self):
        """Test prediction disabled returns raw pose."""
        hmd = HeadMountedDisplay(
            prediction_config=PredictionConfig(enabled=False)
        )

        hmd.update_pose(
            position=(1.0, 1.5, 2.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            linear_velocity=(10.0, 0.0, 0.0),
            confidence=1.0,
        )

        assert hmd.predicted_position == (1.0, 1.5, 2.0)

    def test_custom_prediction_time(self):
        """Test get_predicted_pose with custom time."""
        # Use custom prediction config that allows longer prediction
        prediction_config = PredictionConfig(
            enabled=True,
            max_prediction_time_ms=150.0,  # Allow 100ms prediction
        )
        hmd = HeadMountedDisplay(prediction_config=prediction_config)

        hmd.update_pose(
            position=(0.0, 1.5, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            linear_velocity=(1.0, 0.0, -1.0),
            confidence=1.0,
        )

        pos, orient = hmd.get_predicted_pose(prediction_time_ms=100.0)

        # 100ms at 1m/s = 0.1m
        assert abs(pos[0] - 0.1) < 0.001
        assert abs(pos[2] - (-0.1)) < 0.001


# =============================================================================
# VIEW MATRIX TESTS
# =============================================================================


class TestHMDViewMatrix:
    """Test HMD view matrix computation."""

    def test_identity_view_matrix(self):
        """Test view matrix at identity pose."""
        hmd = HeadMountedDisplay()
        hmd.update_pose(
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            confidence=1.0,
        )

        left = hmd.get_left_view_matrix()
        right = hmd.get_right_view_matrix()

        # Both should be valid 4x4 matrices
        assert len(left) == 4
        assert len(left[0]) == 4
        assert len(right) == 4
        assert len(right[0]) == 4

    def test_view_matrix_caching(self):
        """Test view matrix is cached."""
        hmd = HeadMountedDisplay()
        hmd.update_pose(
            position=(0.0, 1.5, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            confidence=1.0,
        )

        left1 = hmd.get_left_view_matrix()
        left2 = hmd.get_left_view_matrix()

        assert left1 is left2  # Same object (cached)

    def test_view_matrix_invalidation(self):
        """Test view matrix is invalidated on pose update."""
        hmd = HeadMountedDisplay()

        hmd.update_pose(
            position=(0.0, 1.5, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            confidence=1.0,
        )
        left1 = hmd.get_left_view_matrix()

        hmd.update_pose(
            position=(1.0, 1.5, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            confidence=1.0,
        )
        left2 = hmd.get_left_view_matrix()

        # Should be different objects (invalidated)
        assert left1 is not left2

    def test_ipd_offset(self):
        """Test IPD offset in view matrices."""
        hmd = HeadMountedDisplay(
            display_info=HMDDisplayInfo(ipd=0.064)
        )

        hmd.update_pose(
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            confidence=1.0,
        )

        left = hmd.get_left_view_matrix()
        right = hmd.get_right_view_matrix()

        # Translation should differ by IPD
        # At identity orientation, translation is in last column
        left_tx = left[3][0]
        right_tx = right[3][0]

        # Difference should be approximately IPD
        assert abs((right_tx - left_tx) - (-0.064)) < 0.001


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================


class TestHMDSerialization:
    """Test HMD serialization."""

    def test_to_dict(self):
        """Test HMD serializes to dictionary."""
        hmd = HeadMountedDisplay(device_id="test_hmd")
        hmd.update_pose(
            position=(1.0, 1.5, -2.0),
            orientation=(0.0, 0.7071, 0.0, 0.7071),
            linear_velocity=(0.5, 0.0, 0.0),
            confidence=0.95,
        )

        data = hmd.to_dict()

        assert data["device_id"] == "test_hmd"
        assert data["position"] == [1.0, 1.5, -2.0]
        assert data["orientation"] == [0.0, 0.7071, 0.0, 0.7071]
        assert data["tracking_state"] == "TRACKING"
        assert data["confidence"] == 0.95

    def test_from_dict(self):
        """Test HMD deserializes from dictionary."""
        data = {
            "device_id": "restored_hmd",
            "position": [2.0, 1.7, -1.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "linear_velocity": [1.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.1, 0.0],
            "tracking_state": "LIMITED",
            "confidence": 0.75,
            "display_info": {
                "resolution_per_eye": [1920, 1080],
                "refresh_rate": 90.0,
                "field_of_view": [90.0, 90.0],
                "ipd": 0.063,
            },
        }

        hmd = HeadMountedDisplay.from_dict(data)

        assert hmd.device_id == "restored_hmd"
        assert hmd.position == (2.0, 1.7, -1.0)
        assert hmd.tracking_state == HMDTrackingState.LIMITED
        assert hmd.confidence == 0.75

    def test_round_trip(self):
        """Test serialization round-trip."""
        original = HeadMountedDisplay(device_id="round_trip_test")
        original.update_pose(
            position=(1.0, 2.0, 3.0),
            orientation=(0.1, 0.2, 0.3, 0.9),
            linear_velocity=(0.5, 0.0, -0.5),
            confidence=0.88,
        )

        data = original.to_dict()
        restored = HeadMountedDisplay.from_dict(data)

        assert restored.device_id == original.device_id
        assert restored.position == original.position
        # Orientation should be approximately equal
        for i in range(4):
            assert abs(restored.orientation[i] - original.orientation[i]) < 0.001


# =============================================================================
# RECENTER TEST
# =============================================================================


class TestHMDRecenter:
    """Test HMD recenter functionality."""

    def test_recenter(self):
        """Test recenter resets position but keeps height."""
        hmd = HeadMountedDisplay()
        hmd.update_pose(
            position=(5.0, 1.6, -3.0),
            orientation=(0.1, 0.2, 0.1, 0.97),
            confidence=1.0,
        )

        hmd.recenter()

        assert hmd.position[0] == 0.0
        assert hmd.position[1] == 1.6  # Height preserved
        assert hmd.position[2] == 0.0
        assert hmd.orientation == (0.0, 0.0, 0.0, 1.0)


# =============================================================================
# DESCRIPTOR TESTS
# =============================================================================


class TestHMDDescriptors:
    """Test HMD uses Trinity descriptors correctly."""

    def test_dirty_tracking(self):
        """Test dirty tracking on pose update."""
        hmd = HeadMountedDisplay()
        clear_dirty(hmd)

        hmd.position = (1.0, 2.0, 3.0)

        assert is_dirty(hmd, "position")

    def test_clear_dirty(self):
        """Test clearing dirty flags."""
        hmd = HeadMountedDisplay()
        hmd.position = (1.0, 2.0, 3.0)
        hmd.orientation = (0.1, 0.0, 0.0, 0.995)

        clear_dirty(hmd)

        assert not is_dirty(hmd, "position")
        assert not is_dirty(hmd, "orientation")
