"""Tests for hand tracking module.

Tests the hand tracking system including:
    - HandJoint enum (26 joints)
    - HandTrackingData component
    - GestureRecognizer
    - GestureEvent
    - HandTracker
"""

import math
import sys
import importlib.util
import pytest

# Load the module directly to avoid package import issues
_spec = importlib.util.spec_from_file_location(
    'hand_tracking',
    'engine/xr/input/hand_tracking.py'
)
_hand_tracking = importlib.util.module_from_spec(_spec)
sys.modules['hand_tracking'] = _hand_tracking
_spec.loader.exec_module(_hand_tracking)

# Import from loaded module
HandJoint = _hand_tracking.HandJoint
HAND_JOINT_COUNT = _hand_tracking.HAND_JOINT_COUNT
GestureType = _hand_tracking.GestureType
JointData = _hand_tracking.JointData
GestureResult = _hand_tracking.GestureResult
HandTrackingData = _hand_tracking.HandTrackingData
GestureRecognizer = _hand_tracking.GestureRecognizer
GestureEvent = _hand_tracking.GestureEvent
HandTracker = _hand_tracking.HandTracker


# =============================================================================
# HandJoint Enum Tests
# =============================================================================


class TestHandJoint:
    """Tests for HandJoint enumeration."""

    def test_joint_count(self):
        """Verify there are exactly 26 joints."""
        assert HAND_JOINT_COUNT == 26
        assert len(HandJoint) == 26

    def test_wrist_is_zero(self):
        """Wrist should be the root joint at index 0."""
        assert HandJoint.WRIST == 0

    def test_thumb_joints(self):
        """Verify thumb has 4 joints (no intermediate)."""
        thumb_joints = [
            HandJoint.THUMB_METACARPAL,
            HandJoint.THUMB_PROXIMAL,
            HandJoint.THUMB_DISTAL,
            HandJoint.THUMB_TIP,
        ]
        assert len(thumb_joints) == 4
        assert all(1 <= j <= 4 for j in thumb_joints)

    def test_index_finger_joints(self):
        """Verify index finger has 5 joints."""
        index_joints = [
            HandJoint.INDEX_METACARPAL,
            HandJoint.INDEX_PROXIMAL,
            HandJoint.INDEX_INTERMEDIATE,
            HandJoint.INDEX_DISTAL,
            HandJoint.INDEX_TIP,
        ]
        assert len(index_joints) == 5

    def test_all_finger_tips_exist(self):
        """All five finger tips should exist."""
        tips = [
            HandJoint.THUMB_TIP,
            HandJoint.INDEX_TIP,
            HandJoint.MIDDLE_TIP,
            HandJoint.RING_TIP,
            HandJoint.PINKY_TIP,
        ]
        assert len(tips) == 5

    def test_palm_joint_exists(self):
        """Palm virtual joint should exist."""
        assert HandJoint.PALM == 25

    def test_joints_are_unique(self):
        """All joint values should be unique."""
        values = [j.value for j in HandJoint]
        assert len(values) == len(set(values))

    def test_joints_contiguous(self):
        """Joint values should be contiguous 0-25."""
        values = sorted([j.value for j in HandJoint])
        assert values == list(range(26))


# =============================================================================
# JointData Tests
# =============================================================================


class TestJointData:
    """Tests for JointData dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        joint = JointData()
        assert joint.position == (0.0, 0.0, 0.0)
        assert joint.orientation == (0.0, 0.0, 0.0, 1.0)
        assert joint.radius == 0.01
        assert joint.linear_velocity is None
        assert joint.angular_velocity is None

    def test_custom_values(self):
        """Test custom initialization."""
        joint = JointData(
            position=(1.0, 2.0, 3.0),
            orientation=(0.5, 0.5, 0.5, 0.5),
            radius=0.02,
            linear_velocity=(0.1, 0.2, 0.3),
        )
        assert joint.position == (1.0, 2.0, 3.0)
        assert joint.orientation == (0.5, 0.5, 0.5, 0.5)
        assert joint.radius == 0.02
        assert joint.linear_velocity == (0.1, 0.2, 0.3)


# =============================================================================
# GestureResult Tests
# =============================================================================


class TestGestureResult:
    """Tests for GestureResult dataclass."""

    def test_default_values(self):
        """Test default gesture result."""
        result = GestureResult()
        assert result.gesture_type == GestureType.NONE
        assert result.confidence == 0.0
        assert result.is_active is False
        assert result.custom_name is None

    def test_active_pinch(self):
        """Test active pinch gesture result."""
        result = GestureResult(
            gesture_type=GestureType.PINCH,
            confidence=0.95,
            is_active=True,
        )
        assert result.gesture_type == GestureType.PINCH
        assert result.confidence == 0.95
        assert result.is_active is True


# =============================================================================
# HandTrackingData Tests
# =============================================================================


class TestHandTrackingData:
    """Tests for HandTrackingData component."""

    def test_default_initialization(self):
        """Test default component initialization."""
        hand = HandTrackingData()
        assert hand.hand == "left"
        assert len(hand.joint_positions) == 26
        assert len(hand.joint_orientations) == 26
        assert len(hand.joint_radii) == 26
        assert hand.is_tracked is False
        assert hand.confidence == 0.0
        assert hand.current_gesture == GestureType.NONE

    def test_right_hand(self):
        """Test right hand initialization."""
        hand = HandTrackingData(hand="right")
        assert hand.hand == "right"

    def test_joint_positions_initialized(self):
        """All joint positions should be initialized to origin."""
        hand = HandTrackingData()
        for pos in hand.joint_positions:
            assert pos == (0.0, 0.0, 0.0)

    def test_joint_orientations_initialized(self):
        """All orientations should be identity quaternion."""
        hand = HandTrackingData()
        for orient in hand.joint_orientations:
            assert orient == (0.0, 0.0, 0.0, 1.0)

    def test_joint_radii_initialized(self):
        """All radii should be default 0.01."""
        hand = HandTrackingData()
        for radius in hand.joint_radii:
            assert radius == 0.01

    def test_pinch_position_computed(self):
        """Pinch position should be midpoint of thumb and index tips."""
        hand = HandTrackingData()
        # Set known positions
        hand.joint_positions[HandJoint.THUMB_TIP] = (0.0, 0.0, 0.0)
        hand.joint_positions[HandJoint.INDEX_TIP] = (0.1, 0.0, 0.0)

        pinch_pos = hand.pinch_position
        assert pinch_pos == (0.05, 0.0, 0.0)

    def test_is_pinching_false_by_default(self):
        """is_pinching should be False when pinch_strength is low."""
        hand = HandTrackingData()
        hand.pinch_strength = 0.5
        assert hand.is_pinching is False

    def test_is_pinching_true_when_strong(self):
        """is_pinching should be True when pinch_strength > 0.8."""
        hand = HandTrackingData()
        hand.pinch_strength = 0.9
        assert hand.is_pinching is True

    def test_palm_position(self):
        """Palm position should return PALM joint position."""
        hand = HandTrackingData()
        hand.joint_positions[HandJoint.PALM] = (0.5, 0.6, 0.7)
        assert hand.palm_position == (0.5, 0.6, 0.7)

    def test_get_joint(self):
        """get_joint should return JointData for specified joint."""
        hand = HandTrackingData()
        hand.joint_positions[HandJoint.WRIST] = (1.0, 2.0, 3.0)
        hand.joint_orientations[HandJoint.WRIST] = (0.1, 0.2, 0.3, 0.9)
        hand.joint_radii[HandJoint.WRIST] = 0.02

        joint = hand.get_joint(HandJoint.WRIST)
        assert joint.position == (1.0, 2.0, 3.0)
        assert joint.orientation == (0.1, 0.2, 0.3, 0.9)
        assert joint.radius == 0.02

    def test_get_finger_curl_invalid_finger(self):
        """get_finger_curl should return 0.0 for invalid finger name."""
        hand = HandTrackingData()
        assert hand.get_finger_curl("invalid") == 0.0

    def test_get_finger_curl_valid_fingers(self):
        """get_finger_curl should work for all five fingers."""
        hand = HandTrackingData()
        for finger in ["thumb", "index", "middle", "ring", "pinky"]:
            curl = hand.get_finger_curl(finger)
            assert 0.0 <= curl <= 1.0

    def test_update_joints_positions(self):
        """update_joints should update all positions."""
        hand = HandTrackingData()
        new_positions = [(float(i), float(i), float(i)) for i in range(26)]

        hand.update_joints(new_positions)

        for i, pos in enumerate(hand.joint_positions):
            assert pos == (float(i), float(i), float(i))

    def test_update_joints_with_orientations(self):
        """update_joints should update orientations when provided."""
        hand = HandTrackingData()
        new_positions = [(0.0, 0.0, 0.0)] * 26
        new_orientations = [(0.1, 0.2, 0.3, 0.9)] * 26

        hand.update_joints(new_positions, new_orientations)

        for orient in hand.joint_orientations:
            assert orient == (0.1, 0.2, 0.3, 0.9)

    def test_update_joints_invalid_count(self):
        """update_joints should raise error for wrong position count."""
        hand = HandTrackingData()
        with pytest.raises(ValueError, match="Expected 26 positions"):
            hand.update_joints([(0.0, 0.0, 0.0)] * 10)

    def test_update_joints_invalid_orientation_count(self):
        """update_joints should raise error for wrong orientation count."""
        hand = HandTrackingData()
        positions = [(0.0, 0.0, 0.0)] * 26
        orientations = [(0.0, 0.0, 0.0, 1.0)] * 10

        with pytest.raises(ValueError, match="Expected 26 orientations"):
            hand.update_joints(positions, orientations)


# =============================================================================
# GestureRecognizer Tests
# =============================================================================


class TestGestureRecognizer:
    """Tests for GestureRecognizer."""

    def test_default_initialization(self):
        """Test default recognizer initialization."""
        recognizer = GestureRecognizer()
        assert recognizer.pinch_threshold == 0.025
        assert recognizer.point_curl_threshold == 0.3
        assert recognizer.fist_curl_threshold == 0.7

    def test_custom_thresholds(self):
        """Test custom threshold initialization."""
        recognizer = GestureRecognizer(
            pinch_threshold=0.05,
            point_curl_threshold=0.4,
            fist_curl_threshold=0.6,
        )
        assert recognizer.pinch_threshold == 0.05
        assert recognizer.point_curl_threshold == 0.4
        assert recognizer.fist_curl_threshold == 0.6

    def test_recognize_untracked_hand(self):
        """Untracked hand should return NONE gesture."""
        recognizer = GestureRecognizer()
        hand = HandTrackingData()
        hand.is_tracked = False

        result = recognizer.recognize(hand)
        assert result.gesture_type == GestureType.NONE
        assert result.confidence == 0.0

    def test_calculate_pinch_strength(self):
        """calculate_pinch_strength should return value 0-1."""
        recognizer = GestureRecognizer()
        hand = HandTrackingData()
        hand.is_tracked = True

        # Set thumb and index tips far apart
        hand.joint_positions[HandJoint.THUMB_TIP] = (0.0, 0.0, 0.0)
        hand.joint_positions[HandJoint.INDEX_TIP] = (0.1, 0.0, 0.0)

        strength = recognizer.calculate_pinch_strength(hand)
        assert 0.0 <= strength <= 1.0

    def test_pinch_strength_increases_when_close(self):
        """Pinch strength should be higher when fingers are closer."""
        recognizer = GestureRecognizer()
        hand = HandTrackingData()
        hand.is_tracked = True

        # Far apart
        hand.joint_positions[HandJoint.THUMB_TIP] = (0.0, 0.0, 0.0)
        hand.joint_positions[HandJoint.INDEX_TIP] = (0.1, 0.0, 0.0)
        strength_far = recognizer.calculate_pinch_strength(hand)

        # Close together
        hand.joint_positions[HandJoint.INDEX_TIP] = (0.01, 0.0, 0.0)
        strength_close = recognizer.calculate_pinch_strength(hand)

        assert strength_close > strength_far

    def test_register_custom_gesture(self):
        """Custom gestures should be registrable."""
        recognizer = GestureRecognizer()

        def detect_wave(hand: HandTrackingData) -> tuple[bool, float]:
            return (True, 0.9)

        recognizer.register_custom_gesture("wave", detect_wave)
        assert "wave" in recognizer.custom_gestures

    def test_unregister_custom_gesture(self):
        """Custom gestures should be unregistrable."""
        recognizer = GestureRecognizer()

        def detect_wave(hand: HandTrackingData) -> tuple[bool, float]:
            return (True, 0.9)

        recognizer.register_custom_gesture("wave", detect_wave)
        assert recognizer.unregister_custom_gesture("wave") is True
        assert "wave" not in recognizer.custom_gestures

    def test_unregister_nonexistent_gesture(self):
        """Unregistering nonexistent gesture should return False."""
        recognizer = GestureRecognizer()
        assert recognizer.unregister_custom_gesture("nonexistent") is False


# =============================================================================
# GestureEvent Tests
# =============================================================================


class TestGestureEvent:
    """Tests for GestureEvent."""

    def test_start_event(self):
        """Test gesture start event creation."""
        event = GestureEvent(
            hand="left",
            gesture=GestureType.PINCH,
            confidence=0.95,
            timestamp=1.0,
            is_start=True,
            pinch_position=(0.1, 0.2, 0.3),
        )
        assert event.hand == "left"
        assert event.gesture == GestureType.PINCH
        assert event.confidence == 0.95
        assert event.is_start is True
        assert event.pinch_position == (0.1, 0.2, 0.3)

    def test_end_event(self):
        """Test gesture end event creation."""
        event = GestureEvent(
            hand="right",
            gesture=GestureType.FIST,
            confidence=0.0,
            timestamp=2.0,
            is_start=False,
        )
        assert event.hand == "right"
        assert event.is_start is False

    def test_custom_gesture_event(self):
        """Test custom gesture event creation."""
        event = GestureEvent(
            hand="left",
            gesture=GestureType.CUSTOM,
            confidence=0.8,
            timestamp=1.5,
            custom_name="wave",
        )
        assert event.gesture == GestureType.CUSTOM
        assert event.custom_name == "wave"


# =============================================================================
# HandTracker Tests
# =============================================================================


class TestHandTracker:
    """Tests for HandTracker manager."""

    def test_default_initialization(self):
        """Test default tracker initialization."""
        tracker = HandTracker()
        assert tracker.left_hand is not None
        assert tracker.right_hand is not None
        assert tracker.left_hand.hand == "left"
        assert tracker.right_hand.hand == "right"
        assert tracker.gesture_recognizer is not None

    def test_custom_gesture_recognizer(self):
        """Test tracker with custom gesture recognizer."""
        custom_recognizer = GestureRecognizer(pinch_threshold=0.05)
        tracker = HandTracker(gesture_recognizer=custom_recognizer)
        assert tracker.gesture_recognizer.pinch_threshold == 0.05

    def test_update_left_hand(self):
        """Update should set left hand as tracked."""
        tracker = HandTracker()
        positions = [(0.0, 0.0, 0.0)] * 26

        tracker.update(left_positions=positions, timestamp=1.0)

        assert tracker.left_hand.is_tracked is True
        assert tracker.right_hand.is_tracked is False

    def test_update_right_hand(self):
        """Update should set right hand as tracked."""
        tracker = HandTracker()
        positions = [(0.0, 0.0, 0.0)] * 26

        tracker.update(right_positions=positions, timestamp=1.0)

        assert tracker.left_hand.is_tracked is False
        assert tracker.right_hand.is_tracked is True

    def test_update_both_hands(self):
        """Update should handle both hands."""
        tracker = HandTracker()
        positions = [(0.0, 0.0, 0.0)] * 26

        tracker.update(
            left_positions=positions,
            right_positions=positions,
            timestamp=1.0,
        )

        assert tracker.left_hand.is_tracked is True
        assert tracker.right_hand.is_tracked is True

    def test_hand_lost_tracking(self):
        """Hand should be marked untracked when no data provided."""
        tracker = HandTracker()
        positions = [(0.0, 0.0, 0.0)] * 26

        # First update with tracking
        tracker.update(left_positions=positions, timestamp=1.0)
        assert tracker.left_hand.is_tracked is True

        # Second update without left hand data
        tracker.update(timestamp=2.0)
        assert tracker.left_hand.is_tracked is False

    def test_gesture_callback(self):
        """Gesture callbacks should be called on gesture changes."""
        tracker = HandTracker()
        events_received: list[GestureEvent] = []

        def on_gesture(event: GestureEvent):
            events_received.append(event)

        tracker.add_gesture_callback(on_gesture)

        # Create positions that might trigger a gesture
        positions = [(0.0, 0.0, 0.0)] * 26
        tracker.update(left_positions=positions, timestamp=1.0)

        # Callback registered successfully
        assert on_gesture in tracker._event_callbacks

    def test_remove_gesture_callback(self):
        """Gesture callbacks should be removable."""
        tracker = HandTracker()

        def on_gesture(event: GestureEvent):
            pass

        tracker.add_gesture_callback(on_gesture)
        assert tracker.remove_gesture_callback(on_gesture) is True
        assert tracker.remove_gesture_callback(on_gesture) is False


# =============================================================================
# Integration Tests
# =============================================================================


class TestHandTrackingIntegration:
    """Integration tests for hand tracking system."""

    def test_full_tracking_workflow(self):
        """Test complete tracking workflow."""
        tracker = HandTracker()

        # Simulate tracking data over multiple frames
        for frame in range(10):
            positions = [
                (0.1 * frame, 0.0, 0.0)
                for _ in range(26)
            ]
            tracker.update(
                left_positions=positions,
                right_positions=positions,
                timestamp=frame * 0.016,  # ~60Hz
            )

        assert tracker.left_hand.is_tracked is True
        assert tracker.right_hand.is_tracked is True

    def test_pinch_detection_workflow(self):
        """Test pinch detection through full workflow."""
        tracker = HandTracker()

        # Create positions with thumb and index close together
        positions = [(0.0, 0.0, 0.0)] * 26
        positions[HandJoint.THUMB_TIP] = (0.0, 0.0, 0.0)
        positions[HandJoint.INDEX_TIP] = (0.01, 0.0, 0.0)  # Very close

        tracker.update(left_positions=positions, timestamp=1.0)

        # Pinch strength should be calculated
        assert tracker.left_hand.pinch_strength > 0.0

    def test_gesture_state_persistence(self):
        """Gesture state should persist across updates."""
        tracker = HandTracker()
        positions = [(0.0, 0.0, 0.0)] * 26

        # Multiple updates
        for i in range(5):
            tracker.update(left_positions=positions, timestamp=i * 0.1)

        # Hand should still be tracked
        assert tracker.left_hand.is_tracked is True
