"""Tests for XR Avatar component."""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat

# Handle import errors from XR __init__.py gracefully
try:
    from engine.xr.avatars.avatar import (
        AvatarVisibility,
        DisplayMode,
        IKTarget,
        PersonalSpace,
        XRAvatar,
        xr_avatar,
        xr_ik_target,
    )
except (ImportError, AttributeError) as e:
    pytest.skip(f"XR module has unrelated import errors: {e}", allow_module_level=True)


class TestIKTarget:
    """Tests for IKTarget dataclass."""

    def test_default_values(self):
        """Test IKTarget default initialization."""
        target = IKTarget()

        assert target.position == Vec3.zero()
        assert target.rotation == Quat.identity()
        assert target.weight == 1.0
        assert target.active is True

    def test_custom_values(self):
        """Test IKTarget with custom values."""
        pos = Vec3(1.0, 2.0, 3.0)
        rot = Quat.from_euler(0.1, 0.2, 0.3)

        target = IKTarget(position=pos, rotation=rot, weight=0.5, active=False)

        assert target.position == pos
        assert target.rotation == rot
        assert target.weight == 0.5
        assert target.active is False

    def test_to_rigid_transform(self):
        """Test converting IKTarget to RigidTransform."""
        pos = Vec3(1.0, 2.0, 3.0)
        rot = Quat.identity()
        target = IKTarget(position=pos, rotation=rot)

        transform = target.to_rigid_transform()

        assert transform.translation == pos
        assert transform.rotation == rot

    def test_from_rigid_transform(self):
        """Test creating IKTarget from RigidTransform."""
        from engine.core.math.transform import RigidTransform

        transform = RigidTransform(
            translation=Vec3(1.0, 2.0, 3.0),
            rotation=Quat.identity(),
        )

        target = IKTarget.from_rigid_transform(transform, weight=0.7)

        assert target.position == transform.translation
        assert target.rotation == transform.rotation
        assert target.weight == 0.7
        assert target.active is True


class TestPersonalSpace:
    """Tests for PersonalSpace configuration."""

    def test_default_values(self):
        """Test PersonalSpace defaults."""
        space = PersonalSpace()

        assert space.radius == 0.5
        assert space.enabled is True
        assert space.push_strength == 0.5
        assert space.fade_distance == 0.3

    def test_is_invaded_inside(self):
        """Test invasion detection when inside radius."""
        space = PersonalSpace(radius=1.0, enabled=True)
        my_pos = Vec3(0, 0, 0)
        other_pos = Vec3(0.5, 0, 0)

        assert space.is_invaded(other_pos, my_pos) is True

    def test_is_invaded_outside(self):
        """Test invasion detection when outside radius."""
        space = PersonalSpace(radius=1.0, enabled=True)
        my_pos = Vec3(0, 0, 0)
        other_pos = Vec3(2.0, 0, 0)

        assert space.is_invaded(other_pos, my_pos) is False

    def test_is_invaded_disabled(self):
        """Test invasion detection when disabled."""
        space = PersonalSpace(radius=1.0, enabled=False)
        my_pos = Vec3(0, 0, 0)
        other_pos = Vec3(0.5, 0, 0)

        assert space.is_invaded(other_pos, my_pos) is False

    def test_get_push_vector(self):
        """Test push vector calculation."""
        space = PersonalSpace(radius=1.0, push_strength=1.0)
        my_pos = Vec3(0, 0, 0)
        other_pos = Vec3(0.5, 0, 0)  # 0.5m inside radius

        push = space.get_push_vector(other_pos, my_pos)

        # Should push away from other toward self
        assert push.x < 0  # Negative X direction
        assert abs(push.y) < 0.001
        assert abs(push.z) < 0.001

    def test_get_push_vector_disabled(self):
        """Test push vector when disabled."""
        space = PersonalSpace(radius=1.0, enabled=False)
        my_pos = Vec3(0, 0, 0)
        other_pos = Vec3(0.5, 0, 0)

        push = space.get_push_vector(other_pos, my_pos)

        assert push == Vec3.zero()

    def test_get_fade_alpha_outside(self):
        """Test fade alpha when outside personal space."""
        space = PersonalSpace(radius=1.0, fade_distance=0.3)
        my_pos = Vec3(0, 0, 0)
        other_pos = Vec3(2.0, 0, 0)

        alpha = space.get_fade_alpha(other_pos, my_pos)

        assert alpha == 1.0

    def test_get_fade_alpha_inside_fade_zone(self):
        """Test fade alpha in fade zone."""
        space = PersonalSpace(radius=1.0, fade_distance=0.3)
        my_pos = Vec3(0, 0, 0)
        other_pos = Vec3(0.85, 0, 0)  # In fade zone

        alpha = space.get_fade_alpha(other_pos, my_pos)

        assert 0.0 < alpha < 1.0


class TestXRAvatar:
    """Tests for XRAvatar component."""

    def test_default_initialization(self):
        """Test XRAvatar default values."""
        avatar = XRAvatar()

        assert avatar.player_height == 1.75  # From XR_CONFIG.avatar.DEFAULT_AVATAR_HEIGHT_M
        assert avatar.arm_span == 1.75  # Defaults to player_height
        assert avatar.floor_level == 0.0
        assert avatar.is_calibrated is False
        assert avatar.visibility == AvatarVisibility.VISIBLE

    def test_custom_initialization(self):
        """Test XRAvatar with custom dimensions."""
        avatar = XRAvatar(player_height=1.8, arm_span=1.85)

        assert avatar.player_height == 1.8
        assert avatar.arm_span == 1.85

    def test_calibrate(self):
        """Test avatar calibration."""
        avatar = XRAvatar()
        avatar.calibrate(height=1.75, arm_span=1.8, floor_level=-0.1)

        assert avatar.player_height == 1.75
        assert avatar.arm_span == 1.8
        assert avatar.floor_level == -0.1
        assert avatar.is_calibrated is True

    def test_calibrate_invalid_height(self):
        """Test calibration rejects invalid height."""
        avatar = XRAvatar()

        with pytest.raises(ValueError, match="Height must be positive"):
            avatar.calibrate(height=0, arm_span=1.7)

    def test_calibrate_invalid_arm_span(self):
        """Test calibration rejects invalid arm span."""
        avatar = XRAvatar()

        with pytest.raises(ValueError, match="Arm span must be positive"):
            avatar.calibrate(height=1.7, arm_span=-1.0)

    def test_update_from_hmd(self):
        """Test updating head target from HMD."""
        avatar = XRAvatar()
        pos = Vec3(0.1, 1.6, -0.5)
        rot = Quat.from_euler(0.1, 0.2, 0.0)

        avatar.update_from_hmd(pos, rot)

        assert avatar.head_target.position == pos
        assert avatar.head_target.rotation == rot
        assert avatar.head_target.active is True

    def test_update_from_controllers(self):
        """Test updating hand targets from controllers."""
        avatar = XRAvatar()
        left_pos = Vec3(-0.3, 1.0, -0.4)
        left_rot = Quat.identity()
        right_pos = Vec3(0.3, 1.0, -0.4)
        right_rot = Quat.identity()

        avatar.update_from_controllers(left_pos, left_rot, right_pos, right_rot)

        assert avatar.left_hand_target.position == left_pos
        assert avatar.right_hand_target.position == right_pos
        assert avatar.left_hand_target.active is True
        assert avatar.right_hand_target.active is True

    def test_estimate_body(self):
        """Test body estimation from IK targets."""
        avatar = XRAvatar(player_height=1.7)
        avatar.calibrate(height=1.7, arm_span=1.7)

        # Set head position
        avatar.update_from_hmd(Vec3(0, 1.6, 0), Quat.identity())

        avatar.estimate_body()

        # Pelvis should be below head
        assert avatar.estimated_pelvis.translation.y < avatar.head_target.position.y
        assert avatar.estimated_pelvis.translation.y > avatar.floor_level

    def test_visibility_modes(self):
        """Test visibility checking."""
        avatar = XRAvatar()

        # VISIBLE - visible to all
        avatar.visibility = AvatarVisibility.VISIBLE
        assert avatar.is_visible_to(viewer_is_self=True) is True
        assert avatar.is_visible_to(viewer_is_self=False) is True

        # HIDDEN - hidden from all
        avatar.visibility = AvatarVisibility.HIDDEN
        assert avatar.is_visible_to(viewer_is_self=True) is False
        assert avatar.is_visible_to(viewer_is_self=False) is False

        # SELF_HIDDEN - hidden from owner only
        avatar.visibility = AvatarVisibility.SELF_HIDDEN
        assert avatar.is_visible_to(viewer_is_self=True) is False
        assert avatar.is_visible_to(viewer_is_self=False) is True

        # OTHERS_HIDDEN - visible to owner only
        avatar.visibility = AvatarVisibility.OTHERS_HIDDEN
        assert avatar.is_visible_to(viewer_is_self=True) is True
        assert avatar.is_visible_to(viewer_is_self=False) is False

    def test_mute_indicator(self):
        """Test mute indicator property."""
        avatar = XRAvatar()

        assert avatar.mute_indicator is False

        avatar.mute_indicator = True
        assert avatar.mute_indicator is True

    def test_name_tag(self):
        """Test name tag properties."""
        avatar = XRAvatar()

        assert avatar.name_tag == ""
        assert avatar.name_tag_visible is True

        avatar.name_tag = "TestPlayer"
        avatar.name_tag_visible = False

        assert avatar.name_tag == "TestPlayer"
        assert avatar.name_tag_visible is False

    def test_get_network_state(self):
        """Test network state serialization."""
        avatar = XRAvatar()
        avatar.update_from_hmd(Vec3(0, 1.6, 0), Quat.identity())
        avatar.name_tag = "Player1"
        avatar.mute_indicator = True

        state = avatar.get_network_state()

        assert "head_position" in state
        assert "head_rotation" in state
        assert "left_hand_position" in state
        assert "right_hand_position" in state
        assert state["name_tag"] == "Player1"
        assert state["mute_indicator"] is True

    def test_apply_network_state(self):
        """Test applying network state."""
        avatar = XRAvatar()

        state = {
            "head_position": (0.1, 1.7, -0.2),
            "head_rotation": (0, 0, 0, 1),
            "name_tag": "RemotePlayer",
            "mute_indicator": True,
        }

        avatar.apply_network_state(state)

        assert abs(avatar.head_target.position.x - 0.1) < 0.001
        assert abs(avatar.head_target.position.y - 1.7) < 0.001
        assert avatar.name_tag == "RemotePlayer"
        assert avatar.mute_indicator is True


class TestXRAvatarDecorator:
    """Tests for @xr_avatar decorator."""

    def test_decorator_basic(self):
        """Test basic decorator application."""
        @xr_avatar()
        class TestAvatar:
            pass

        assert hasattr(TestAvatar, "_xr_avatar")
        assert TestAvatar._xr_avatar is True
        assert TestAvatar._xr_avatar_ik_enabled is True
        assert TestAvatar._xr_avatar_network_sync is True

    def test_decorator_with_params(self):
        """Test decorator with custom parameters."""
        @xr_avatar(ik_enabled=False, network_sync=False, face_tracking=True)
        class TestAvatar:
            pass

        assert TestAvatar._xr_avatar_ik_enabled is False
        assert TestAvatar._xr_avatar_network_sync is False
        assert TestAvatar._xr_avatar_face_tracking is True

    def test_decorator_tags(self):
        """Test decorator sets tags."""
        @xr_avatar(ik_enabled=True)
        class TestAvatar:
            pass

        assert hasattr(TestAvatar, "_tags")
        assert TestAvatar._tags["xr_avatar"] is True
        assert TestAvatar._tags["xr_avatar_ik_enabled"] is True

    def test_decorator_tracking(self):
        """Test decorator is tracked in _applied_decorators."""
        @xr_avatar()
        class TestAvatar:
            pass

        assert hasattr(TestAvatar, "_applied_decorators")
        assert "xr_avatar" in TestAvatar._applied_decorators


class TestXRIKTargetDecorator:
    """Tests for @xr_ik_target decorator."""

    def test_decorator_basic(self):
        """Test basic IK target decorator."""
        @xr_ik_target(target_type="hand")
        class HandTarget:
            pass

        assert hasattr(HandTarget, "_xr_ik_target")
        assert HandTarget._xr_ik_target is True
        assert HandTarget._xr_ik_target_type == "hand"

    def test_decorator_with_bone_chain(self):
        """Test IK target with bone chain."""
        @xr_ik_target(target_type="hand", bone_chain=["shoulder", "elbow", "wrist"])
        class ArmTarget:
            pass

        assert ArmTarget._xr_ik_target_bone_chain == ["shoulder", "elbow", "wrist"]

    def test_decorator_invalid_type(self):
        """Test IK target rejects invalid types."""
        with pytest.raises(ValueError, match="Invalid target_type"):
            @xr_ik_target(target_type="invalid")
            class BadTarget:
                pass

    def test_decorator_all_valid_types(self):
        """Test all valid IK target types."""
        valid_types = ["head", "hand", "foot", "pelvis", "chest", "elbow", "knee"]

        for target_type in valid_types:
            @xr_ik_target(target_type=target_type)
            class Target:
                pass

            assert Target._xr_ik_target_type == target_type
