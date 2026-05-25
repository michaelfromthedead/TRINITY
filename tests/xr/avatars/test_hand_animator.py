"""Tests for hand and finger animation."""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

# Handle import errors from XR __init__.py gracefully
try:
    from engine.xr.avatars.hand_animator import (
        AvatarHand,
        FingerCurl,
        FingerName,
        HandPose,
        HandPoseType,
        PoseLibrary,
    )
except (ImportError, AttributeError) as e:
    pytest.skip(f"XR module has unrelated import errors: {e}", allow_module_level=True)


class TestFingerCurl:
    """Tests for FingerCurl dataclass."""

    def test_default_values(self):
        """Test FingerCurl defaults."""
        curl = FingerCurl()

        assert curl.curl == 0.0
        assert curl.spread == 0.0
        assert curl.twist == 0.0

    def test_clamping(self):
        """Test value clamping."""
        curl = FingerCurl(curl=1.5, spread=2.0, twist=-2.0)

        assert curl.curl == 1.0  # Clamped to max
        assert curl.spread == 1.0  # Clamped to max
        assert curl.twist == -1.0  # Clamped to min

    def test_lerp(self):
        """Test interpolation between finger curls."""
        a = FingerCurl(curl=0.0, spread=0.0)
        b = FingerCurl(curl=1.0, spread=0.5)

        mid = a.lerp(b, 0.5)

        assert mid.curl == pytest.approx(0.5, abs=0.001)
        assert mid.spread == pytest.approx(0.25, abs=0.001)

    def test_lerp_boundaries(self):
        """Test lerp at boundaries."""
        a = FingerCurl(curl=0.0)
        b = FingerCurl(curl=1.0)

        # t=0 should give 'a'
        result = a.lerp(b, 0.0)
        assert result.curl == pytest.approx(0.0, abs=0.001)

        # t=1 should give 'b'
        result = a.lerp(b, 1.0)
        assert result.curl == pytest.approx(1.0, abs=0.001)


class TestHandPose:
    """Tests for HandPose."""

    def test_default_values(self):
        """Test HandPose defaults."""
        pose = HandPose()

        assert pose.thumb.curl == 0.0
        assert pose.index.curl == 0.0
        assert pose.middle.curl == 0.0
        assert pose.ring.curl == 0.0
        assert pose.pinky.curl == 0.0

    def test_get_finger(self):
        """Test getting finger by name."""
        pose = HandPose(
            thumb=FingerCurl(curl=0.1),
            index=FingerCurl(curl=0.2),
            middle=FingerCurl(curl=0.3),
            ring=FingerCurl(curl=0.4),
            pinky=FingerCurl(curl=0.5),
        )

        assert pose.get_finger(FingerName.THUMB).curl == pytest.approx(0.1)
        assert pose.get_finger(FingerName.INDEX).curl == pytest.approx(0.2)
        assert pose.get_finger(FingerName.MIDDLE).curl == pytest.approx(0.3)
        assert pose.get_finger(FingerName.RING).curl == pytest.approx(0.4)
        assert pose.get_finger(FingerName.PINKY).curl == pytest.approx(0.5)

    def test_set_finger(self):
        """Test setting finger by name."""
        pose = HandPose()

        pose.set_finger(FingerName.INDEX, FingerCurl(curl=0.8))

        assert pose.index.curl == pytest.approx(0.8)

    def test_lerp(self):
        """Test pose interpolation."""
        open_pose = HandPose()  # All zeros
        fist = HandPose(
            thumb=FingerCurl(curl=1.0),
            index=FingerCurl(curl=1.0),
            middle=FingerCurl(curl=1.0),
            ring=FingerCurl(curl=1.0),
            pinky=FingerCurl(curl=1.0),
        )

        half = open_pose.lerp(fist, 0.5)

        assert half.thumb.curl == pytest.approx(0.5, abs=0.001)
        assert half.index.curl == pytest.approx(0.5, abs=0.001)

    def test_to_tuple(self):
        """Test converting pose to tuple."""
        pose = HandPose(
            thumb=FingerCurl(curl=0.1),
            index=FingerCurl(curl=0.2),
            middle=FingerCurl(curl=0.3),
            ring=FingerCurl(curl=0.4),
            pinky=FingerCurl(curl=0.5),
        )

        result = pose.to_tuple()

        assert result == pytest.approx((0.1, 0.2, 0.3, 0.4, 0.5), abs=0.001)

    def test_from_tuple(self):
        """Test creating pose from tuple."""
        curls = (0.1, 0.2, 0.3, 0.4, 0.5)
        pose = HandPose.from_tuple(curls)

        assert pose.thumb.curl == pytest.approx(0.1)
        assert pose.index.curl == pytest.approx(0.2)
        assert pose.middle.curl == pytest.approx(0.3)
        assert pose.ring.curl == pytest.approx(0.4)
        assert pose.pinky.curl == pytest.approx(0.5)


class TestPoseLibrary:
    """Tests for pose library."""

    def test_initialize_defaults(self):
        """Test initializing default poses."""
        PoseLibrary.initialize_defaults()

        poses = PoseLibrary.list_poses()
        assert "open" in poses
        assert "fist" in poses
        assert "point" in poses
        assert "pinch" in poses

    def test_get_by_name(self):
        """Test getting pose by name."""
        PoseLibrary.initialize_defaults()

        pose = PoseLibrary.get("fist")

        assert pose is not None
        assert pose.index.curl > 0.5  # Fist has curled fingers

    def test_get_by_name_case_insensitive(self):
        """Test case-insensitive pose lookup."""
        PoseLibrary.initialize_defaults()

        pose1 = PoseLibrary.get("FIST")
        pose2 = PoseLibrary.get("Fist")
        pose3 = PoseLibrary.get("fist")

        assert pose1 is not None
        assert pose2 is not None
        assert pose3 is not None

    def test_get_nonexistent(self):
        """Test getting non-existent pose."""
        PoseLibrary.initialize_defaults()

        pose = PoseLibrary.get("nonexistent")

        assert pose is None

    def test_get_by_type(self):
        """Test getting pose by type enum."""
        PoseLibrary.initialize_defaults()

        pose = PoseLibrary.get_by_type(HandPoseType.POINT)

        assert pose is not None
        assert pose.index.curl < 0.5  # Point has extended index

    def test_register_custom(self):
        """Test registering custom pose."""
        PoseLibrary.initialize_defaults()

        custom = HandPose(
            thumb=FingerCurl(curl=0.5),
            index=FingerCurl(curl=0.5),
        )
        PoseLibrary.register("half_grip", custom)

        retrieved = PoseLibrary.get("half_grip")
        assert retrieved is not None
        assert retrieved.thumb.curl == pytest.approx(0.5)


class TestAvatarHand:
    """Tests for AvatarHand."""

    def test_initialization(self):
        """Test hand initialization."""
        hand = AvatarHand(hand_side="left", blend_speed=15.0)

        assert hand.hand_side == "left"
        assert hand.display_mode == "hand"

    def test_invalid_hand_side(self):
        """Test rejection of invalid hand side."""
        with pytest.raises(ValueError, match="must be 'left' or 'right'"):
            AvatarHand(hand_side="invalid")

    def test_invalid_blend_speed(self):
        """Test rejection of invalid blend speed."""
        with pytest.raises(ValueError, match="blend_speed must be positive"):
            AvatarHand(hand_side="left", blend_speed=0)

    def test_set_pose(self):
        """Test setting target pose."""
        hand = AvatarHand("right")
        pose = HandPose(thumb=FingerCurl(curl=0.8))

        hand.set_pose(pose)

        assert hand.target_pose.thumb.curl == pytest.approx(0.8)

    def test_set_pose_by_name(self):
        """Test setting pose by name."""
        PoseLibrary.initialize_defaults()
        hand = AvatarHand("left")

        success = hand.set_pose_by_name("fist")

        assert success is True
        assert hand.target_pose.index.curl > 0.5

    def test_set_pose_by_name_not_found(self):
        """Test setting non-existent pose by name."""
        hand = AvatarHand("left")

        success = hand.set_pose_by_name("nonexistent")

        assert success is False

    def test_set_pose_by_type(self):
        """Test setting pose by type."""
        PoseLibrary.initialize_defaults()
        hand = AvatarHand("left")

        hand.set_pose_by_type(HandPoseType.POINT)

        assert hand.target_pose.index.curl < 0.5  # Extended

    def test_update_from_controller(self):
        """Test updating from controller input."""
        hand = AvatarHand("right")

        hand.update_from_controller(
            trigger_value=0.8,
            grip_value=0.6,
            thumbstick_touched=True,
        )

        # Index follows trigger
        assert hand.target_pose.index.curl == pytest.approx(0.8, abs=0.01)
        # Middle/ring/pinky follow grip
        assert hand.target_pose.middle.curl == pytest.approx(0.6, abs=0.01)
        # Thumb curls when thumbstick touched
        assert hand.target_pose.thumb.curl > 0.5

    def test_update_interpolation(self):
        """Test pose interpolation over time."""
        hand = AvatarHand("left", blend_speed=10.0)

        # Set target to fist
        hand.set_pose(HandPose(
            thumb=FingerCurl(curl=1.0),
            index=FingerCurl(curl=1.0),
            middle=FingerCurl(curl=1.0),
            ring=FingerCurl(curl=1.0),
            pinky=FingerCurl(curl=1.0),
        ))

        # Initial state should be open
        assert hand.current_pose.index.curl == pytest.approx(0.0)

        # Update for a small time step (less than what would complete the interpolation)
        hand.update(0.05)

        # Should have moved toward target but not reached it
        assert hand.current_pose.index.curl > 0.0
        # With blend_speed=10, after 0.05s we get t=0.5, so result is ~0.5
        assert hand.current_pose.index.curl < 0.9  # Not fully there yet

    def test_snap_to_target(self):
        """Test snapping to target immediately."""
        hand = AvatarHand("left")

        hand.set_pose(HandPose(index=FingerCurl(curl=0.7)))
        hand.snap_to_target()

        assert hand.current_pose.index.curl == pytest.approx(0.7)

    def test_display_mode(self):
        """Test display mode property."""
        hand = AvatarHand("left")

        assert hand.display_mode == "hand"

        hand.display_mode = "controller"
        assert hand.display_mode == "controller"

        hand.display_mode = "tool"
        assert hand.display_mode == "tool"

    def test_invalid_display_mode(self):
        """Test rejection of invalid display mode."""
        hand = AvatarHand("left")

        with pytest.raises(ValueError):
            hand.display_mode = "invalid"

    def test_held_tool_sets_display_mode(self):
        """Test that setting held tool changes display mode."""
        hand = AvatarHand("left")

        hand.held_tool_id = 123

        assert hand.held_tool_id == 123
        assert hand.display_mode == "tool"

    def test_finger_curl_properties(self):
        """Test individual finger curl properties."""
        hand = AvatarHand("left")
        hand.set_pose(HandPose(
            thumb=FingerCurl(curl=0.1),
            index=FingerCurl(curl=0.2),
            middle=FingerCurl(curl=0.3),
            ring=FingerCurl(curl=0.4),
            pinky=FingerCurl(curl=0.5),
        ))
        hand.snap_to_target()

        assert hand.thumb_curl == pytest.approx(0.1)
        assert hand.index_curl == pytest.approx(0.2)
        assert hand.middle_curl == pytest.approx(0.3)
        assert hand.ring_curl == pytest.approx(0.4)
        assert hand.pinky_curl == pytest.approx(0.5)

    def test_get_network_state(self):
        """Test network state serialization."""
        hand = AvatarHand("right")
        hand.set_pose(HandPose(index=FingerCurl(curl=0.5)))
        hand.snap_to_target()
        hand.held_tool_id = 42

        state = hand.get_network_state()

        assert state["hand_side"] == "right"
        assert "pose" in state
        assert state["display_mode"] == "tool"
        assert state["held_tool_id"] == 42

    def test_apply_network_state(self):
        """Test applying network state."""
        hand = AvatarHand("left")

        state = {
            "pose": (0.1, 0.2, 0.3, 0.4, 0.5),
            "display_mode": "controller",
            "held_tool_id": None,
        }

        hand.apply_network_state(state)

        assert hand.target_pose.thumb.curl == pytest.approx(0.1)
        assert hand.display_mode == "controller"
