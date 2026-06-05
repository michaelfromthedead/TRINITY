"""
Whitebox tests for engine/simulation/character/platform_handling.py

Tests PlatformHandler, PlatformData, PlatformAttachment, and platform physics.
"""

import math
import pytest
from engine.simulation.character.platform_handling import (
    AttachmentMode,
    PlatformAttachment,
    PlatformData,
    PlatformHandler,
    PlatformProvider,
    PlatformType,
)
from engine.simulation.character.character_controller import (
    Quaternion,
    Transform,
    Vector3,
)


class TestPlatformType:
    """Tests for PlatformType enum."""

    def test_linear_value(self):
        """LINEAR should have expected value."""
        assert PlatformType.LINEAR.value == "linear"

    def test_rotating_value(self):
        """ROTATING should have expected value."""
        assert PlatformType.ROTATING.value == "rotating"

    def test_elevator_value(self):
        """ELEVATOR should have expected value."""
        assert PlatformType.ELEVATOR.value == "elevator"

    def test_path_value(self):
        """PATH should have expected value."""
        assert PlatformType.PATH.value == "path"

    def test_physics_value(self):
        """PHYSICS should have expected value."""
        assert PlatformType.PHYSICS.value == "physics"


class TestAttachmentMode:
    """Tests for AttachmentMode enum."""

    def test_parent_value(self):
        """PARENT should have expected value."""
        assert AttachmentMode.PARENT.value == "parent"

    def test_velocity_value(self):
        """VELOCITY should have expected value."""
        assert AttachmentMode.VELOCITY.value == "velocity"

    def test_hybrid_value(self):
        """HYBRID should have expected value."""
        assert AttachmentMode.HYBRID.value == "hybrid"


class TestPlatformData:
    """Tests for PlatformData dataclass."""

    def test_default_construction(self):
        """Default PlatformData should have reasonable defaults."""
        data = PlatformData()
        assert data.platform_id == 0
        assert data.platform_type == PlatformType.LINEAR
        assert data.is_active is True

    def test_custom_construction(self):
        """PlatformData should accept custom values."""
        data = PlatformData(
            platform_id=42,
            platform_type=PlatformType.ROTATING,
            velocity=Vector3(1.0, 0.0, 0.0),
            angular_velocity=Vector3(0.0, 1.0, 0.0),
            is_active=True,
        )
        assert data.platform_id == 42
        assert data.platform_type == PlatformType.ROTATING
        assert data.velocity.x == 1.0
        assert data.angular_velocity.y == 1.0


class TestPlatformAttachment:
    """Tests for PlatformAttachment dataclass."""

    def test_default_construction(self):
        """Default PlatformAttachment should have reasonable defaults."""
        attachment = PlatformAttachment()
        assert attachment.platform_id == 0
        assert attachment.attachment_mode == AttachmentMode.HYBRID
        assert attachment.attachment_time == 0.0

    def test_custom_construction(self):
        """PlatformAttachment should accept custom values."""
        attachment = PlatformAttachment(
            platform_id=123,
            local_offset=Vector3(0.5, 0.0, 0.0),
            attachment_mode=AttachmentMode.VELOCITY,
            attachment_time=10.5,
        )
        assert attachment.platform_id == 123
        assert attachment.local_offset.x == 0.5
        assert attachment.attachment_mode == AttachmentMode.VELOCITY


class TestPlatformProvider:
    """Tests for PlatformProvider base class."""

    def test_get_platform_default(self):
        """Default get_platform should return None."""
        provider = PlatformProvider()
        assert provider.get_platform(1) is None

    def test_get_platform_transform_default(self):
        """Default get_platform_transform should return None."""
        provider = PlatformProvider()
        assert provider.get_platform_transform(1) is None

    def test_get_platform_velocity_default(self):
        """Default get_platform_velocity should return zero."""
        provider = PlatformProvider()
        vel = provider.get_platform_velocity(1)
        assert vel.x == 0.0
        assert vel.y == 0.0
        assert vel.z == 0.0

    def test_get_platform_angular_velocity_default(self):
        """Default get_platform_angular_velocity should return zero."""
        provider = PlatformProvider()
        vel = provider.get_platform_angular_velocity(1)
        assert vel.x == 0.0
        assert vel.y == 0.0
        assert vel.z == 0.0


class MockPlatformProvider(PlatformProvider):
    """Mock platform provider for testing."""

    def __init__(self):
        self.platforms: dict[int, PlatformData] = {}

    def get_platform(self, platform_id: int):
        return self.platforms.get(platform_id)

    def get_platform_transform(self, platform_id: int):
        platform = self.platforms.get(platform_id)
        return platform.transform if platform else None

    def get_platform_velocity(self, platform_id: int) -> Vector3:
        platform = self.platforms.get(platform_id)
        return platform.velocity if platform else Vector3.zero()

    def get_platform_angular_velocity(self, platform_id: int) -> Vector3:
        platform = self.platforms.get(platform_id)
        return platform.angular_velocity if platform else Vector3.zero()


class TestPlatformHandler:
    """Tests for PlatformHandler class."""

    def test_construction(self):
        """PlatformHandler should be constructible."""
        provider = MockPlatformProvider()
        handler = PlatformHandler(provider)
        assert handler is not None
        assert handler.is_attached is False

    def test_is_attached_false(self):
        """is_attached should return False when not attached."""
        provider = MockPlatformProvider()
        handler = PlatformHandler(provider)
        assert handler.is_attached is False

    def test_attached_platform_id_none(self):
        """attached_platform_id should return None when not attached."""
        provider = MockPlatformProvider()
        handler = PlatformHandler(provider)
        assert handler.attached_platform_id is None

    def test_attachment_none(self):
        """attachment should return None when not attached."""
        provider = MockPlatformProvider()
        handler = PlatformHandler(provider)
        assert handler.attachment is None

    def test_inherited_velocity_none(self):
        """inherited_velocity should return zero when not attached."""
        provider = MockPlatformProvider()
        handler = PlatformHandler(provider)
        vel = handler.inherited_velocity
        assert vel.x == 0.0
        assert vel.y == 0.0
        assert vel.z == 0.0

    def test_attach_to_platform_success(self):
        """attach_to_platform should succeed with valid platform."""
        provider = MockPlatformProvider()
        provider.platforms[1] = PlatformData(
            platform_id=1,
            transform=Transform(position=Vector3(10.0, 0.0, 0.0)),
        )
        handler = PlatformHandler(provider)
        result = handler.attach_to_platform(
            platform_id=1,
            character_position=Vector3(11.0, 0.0, 0.0),
            character_rotation=Quaternion.identity(),
        )
        assert result is True
        assert handler.is_attached is True
        assert handler.attached_platform_id == 1

    def test_attach_to_platform_invalid(self):
        """attach_to_platform should fail with invalid platform."""
        provider = MockPlatformProvider()
        handler = PlatformHandler(provider)
        result = handler.attach_to_platform(
            platform_id=999,
            character_position=Vector3.zero(),
            character_rotation=Quaternion.identity(),
        )
        assert result is False
        assert handler.is_attached is False

    def test_attach_callback(self):
        """attach_to_platform should trigger callback."""
        provider = MockPlatformProvider()
        provider.platforms[1] = PlatformData(platform_id=1)
        handler = PlatformHandler(provider)
        attached_ids = []
        handler.set_attach_callback(lambda pid: attached_ids.append(pid))
        handler.attach_to_platform(1, Vector3.zero(), Quaternion.identity())
        assert 1 in attached_ids

    def test_detach_from_platform(self):
        """detach_from_platform should clear attachment."""
        provider = MockPlatformProvider()
        provider.platforms[1] = PlatformData(platform_id=1)
        handler = PlatformHandler(provider)
        handler.attach_to_platform(1, Vector3.zero(), Quaternion.identity())
        exit_vel = handler.detach_from_platform()
        assert handler.is_attached is False
        assert handler.attached_platform_id is None

    def test_detach_from_platform_preserves_velocity(self):
        """detach_from_platform should preserve platform velocity."""
        provider = MockPlatformProvider()
        provider.platforms[1] = PlatformData(
            platform_id=1,
            velocity=Vector3(5.0, 0.0, 0.0),
        )
        handler = PlatformHandler(provider)
        handler.attach_to_platform(1, Vector3.zero(), Quaternion.identity())
        exit_vel = handler.detach_from_platform(preserve_velocity=True)
        assert exit_vel.x == 5.0

    def test_detach_callback(self):
        """detach_from_platform should trigger callback."""
        provider = MockPlatformProvider()
        provider.platforms[1] = PlatformData(
            platform_id=1,
            velocity=Vector3(3.0, 0.0, 0.0),
        )
        handler = PlatformHandler(provider)
        handler.attach_to_platform(1, Vector3.zero(), Quaternion.identity())
        detach_data = []
        handler.set_detach_callback(lambda pid, vel: detach_data.append((pid, vel)))
        handler.detach_from_platform()
        assert len(detach_data) == 1
        assert detach_data[0][0] == 1

    def test_detach_when_not_attached(self):
        """detach_from_platform when not attached should return zero."""
        provider = MockPlatformProvider()
        handler = PlatformHandler(provider)
        exit_vel = handler.detach_from_platform()
        assert exit_vel.x == 0.0
        assert exit_vel.y == 0.0
        assert exit_vel.z == 0.0

    def test_get_platform_velocity_not_attached(self):
        """get_platform_velocity should return zero when not attached."""
        provider = MockPlatformProvider()
        handler = PlatformHandler(provider)
        vel = handler.get_platform_velocity()
        assert vel.x == 0.0

    def test_get_platform_velocity_attached(self):
        """get_platform_velocity should return platform velocity."""
        provider = MockPlatformProvider()
        provider.platforms[1] = PlatformData(
            platform_id=1,
            velocity=Vector3(10.0, 0.0, 0.0),
        )
        handler = PlatformHandler(provider)
        handler.attach_to_platform(1, Vector3.zero(), Quaternion.identity())
        vel = handler.get_platform_velocity()
        assert vel.x == 10.0

    def test_get_platform_velocity_clamped(self):
        """get_platform_velocity should clamp extreme velocities."""
        provider = MockPlatformProvider()
        provider.platforms[1] = PlatformData(
            platform_id=1,
            velocity=Vector3(1000.0, 0.0, 0.0),  # Very fast
        )
        handler = PlatformHandler(provider)
        handler.attach_to_platform(1, Vector3.zero(), Quaternion.identity())
        vel = handler.get_platform_velocity()
        assert vel.magnitude() <= handler._max_velocity

    def test_get_point_velocity_not_attached(self):
        """get_point_velocity should return zero when not attached."""
        provider = MockPlatformProvider()
        handler = PlatformHandler(provider)
        vel = handler.get_point_velocity(Vector3(1.0, 0.0, 0.0))
        assert vel.x == 0.0

    def test_get_point_velocity_linear(self):
        """get_point_velocity for linear platform should return linear velocity."""
        provider = MockPlatformProvider()
        provider.platforms[1] = PlatformData(
            platform_id=1,
            velocity=Vector3(5.0, 0.0, 0.0),
        )
        handler = PlatformHandler(provider)
        handler.attach_to_platform(1, Vector3.zero(), Quaternion.identity())
        vel = handler.get_point_velocity(Vector3.zero())
        assert vel.x == 5.0

    def test_get_point_velocity_rotating(self):
        """get_point_velocity for rotating platform should include tangential."""
        provider = MockPlatformProvider()
        provider.platforms[1] = PlatformData(
            platform_id=1,
            velocity=Vector3.zero(),
            angular_velocity=Vector3(0.0, 1.0, 0.0),  # Rotating around Y
            transform=Transform(position=Vector3.zero()),
        )
        handler = PlatformHandler(provider)
        handler.attach_to_platform(1, Vector3(1.0, 0.0, 0.0), Quaternion.identity())
        vel = handler.get_point_velocity(Vector3(1.0, 0.0, 0.0))
        # Tangential velocity should be non-zero
        assert vel.magnitude() > 0


class TestPlatformUpdate:
    """Tests for platform update mechanics."""

    def test_update_not_attached(self):
        """update when not attached should return unchanged values."""
        provider = MockPlatformProvider()
        handler = PlatformHandler(provider)
        pos = Vector3(1.0, 2.0, 3.0)
        rot = Quaternion.identity()
        new_pos, new_rot, vel = handler.update(pos, rot, 0.1)
        assert new_pos.x == pos.x
        assert new_pos.y == pos.y
        assert new_pos.z == pos.z
        assert vel.x == 0.0

    def test_update_platform_removed(self):
        """update should detach if platform no longer exists."""
        provider = MockPlatformProvider()
        provider.platforms[1] = PlatformData(platform_id=1)
        handler = PlatformHandler(provider)
        handler.attach_to_platform(1, Vector3.zero(), Quaternion.identity())
        # Remove platform
        del provider.platforms[1]
        pos = Vector3.zero()
        rot = Quaternion.identity()
        handler.update(pos, rot, 0.1)
        assert handler.is_attached is False

    def test_update_parent_mode(self):
        """update in PARENT mode should follow platform transform."""
        provider = MockPlatformProvider()
        provider.platforms[1] = PlatformData(
            platform_id=1,
            transform=Transform(position=Vector3(10.0, 0.0, 0.0)),
        )
        handler = PlatformHandler(provider)
        handler.attach_to_platform(
            1, Vector3(11.0, 0.0, 0.0), Quaternion.identity(),
            mode=AttachmentMode.PARENT
        )
        # Move platform
        provider.platforms[1].transform.position = Vector3(20.0, 0.0, 0.0)
        new_pos, new_rot, vel = handler.update(
            Vector3(11.0, 0.0, 0.0), Quaternion.identity(), 0.1
        )
        # Character should follow
        assert new_pos.x > 11.0

    def test_update_velocity_mode(self):
        """update in VELOCITY mode should add platform velocity."""
        provider = MockPlatformProvider()
        provider.platforms[1] = PlatformData(
            platform_id=1,
            velocity=Vector3(10.0, 0.0, 0.0),
            transform=Transform(),
        )
        handler = PlatformHandler(provider)
        handler.attach_to_platform(
            1, Vector3.zero(), Quaternion.identity(),
            mode=AttachmentMode.VELOCITY
        )
        new_pos, new_rot, vel = handler.update(
            Vector3.zero(), Quaternion.identity(), 0.1
        )
        # Position should have moved based on velocity
        assert new_pos.x == 1.0  # 10.0 * 0.1

    def test_update_updates_inherited_velocity(self):
        """update should store inherited velocity."""
        provider = MockPlatformProvider()
        provider.platforms[1] = PlatformData(
            platform_id=1,
            velocity=Vector3(5.0, 0.0, 0.0),
            transform=Transform(),
        )
        handler = PlatformHandler(provider)
        handler.attach_to_platform(1, Vector3.zero(), Quaternion.identity())
        handler.update(Vector3.zero(), Quaternion.identity(), 0.1)
        assert handler.inherited_velocity.x == 5.0


class TestRotatingPlatform:
    """Tests for rotating platform handling."""

    def test_handle_rotating_platform_not_attached(self):
        """handle_rotating_platform when not attached should return unchanged."""
        provider = MockPlatformProvider()
        handler = PlatformHandler(provider)
        pos = Vector3(1.0, 0.0, 0.0)
        forward = Vector3.forward()
        new_pos, yaw = handler.handle_rotating_platform(pos, forward, 0.1)
        assert new_pos.x == pos.x
        assert yaw == 0.0

    def test_handle_rotating_platform_not_rotating(self):
        """handle_rotating_platform for non-rotating platform should return unchanged."""
        provider = MockPlatformProvider()
        provider.platforms[1] = PlatformData(
            platform_id=1,
            platform_type=PlatformType.LINEAR,  # Not rotating
        )
        handler = PlatformHandler(provider)
        handler.attach_to_platform(1, Vector3(1.0, 0.0, 0.0), Quaternion.identity())
        pos = Vector3(1.0, 0.0, 0.0)
        new_pos, yaw = handler.handle_rotating_platform(pos, Vector3.forward(), 0.1)
        assert new_pos.x == 1.0
        assert yaw == 0.0

    def test_handle_rotating_platform_no_angular_velocity(self):
        """handle_rotating_platform with no rotation should return unchanged."""
        provider = MockPlatformProvider()
        provider.platforms[1] = PlatformData(
            platform_id=1,
            platform_type=PlatformType.ROTATING,
            angular_velocity=Vector3.zero(),
        )
        handler = PlatformHandler(provider)
        handler.attach_to_platform(1, Vector3(1.0, 0.0, 0.0), Quaternion.identity())
        pos = Vector3(1.0, 0.0, 0.0)
        new_pos, yaw = handler.handle_rotating_platform(pos, Vector3.forward(), 0.1)
        assert yaw == 0.0

    def test_handle_rotating_platform_rotates_position(self):
        """handle_rotating_platform should rotate character position."""
        provider = MockPlatformProvider()
        provider.platforms[1] = PlatformData(
            platform_id=1,
            platform_type=PlatformType.ROTATING,
            angular_velocity=Vector3(0.0, math.pi, 0.0),  # 180 deg/sec around Y
            transform=Transform(position=Vector3.zero()),
        )
        handler = PlatformHandler(provider)
        handler.attach_to_platform(1, Vector3(1.0, 0.0, 0.0), Quaternion.identity())
        # Character at (1, 0, 0) rotating around origin
        pos = Vector3(1.0, 0.0, 0.0)
        new_pos, yaw = handler.handle_rotating_platform(pos, Vector3.forward(), 0.5)
        # After 0.5 sec at pi rad/sec = 90 degrees rotation
        # (1, 0, 0) should be near (0, 0, 1) or (0, 0, -1) depending on direction
        assert abs(new_pos.x) < 0.1 or abs(new_pos.z) > 0.9

    def test_rotate_point_around_axis_identity(self):
        """_rotate_point_around_axis with zero angle should not change point."""
        provider = MockPlatformProvider()
        handler = PlatformHandler(provider)
        point = Vector3(1.0, 0.0, 0.0)
        axis = Vector3.up()
        result = handler._rotate_point_around_axis(point, axis, 0.0)
        assert result.x == pytest.approx(1.0)
        assert result.y == pytest.approx(0.0)
        assert result.z == pytest.approx(0.0)

    def test_rotate_point_around_axis_90_degrees(self):
        """_rotate_point_around_axis should rotate point correctly."""
        provider = MockPlatformProvider()
        handler = PlatformHandler(provider)
        point = Vector3(1.0, 0.0, 0.0)
        axis = Vector3.up()
        result = handler._rotate_point_around_axis(point, axis, math.pi / 2)
        # Rotated 90 degrees around Y: (1,0,0) -> (0,0,-1) or (0,0,1)
        assert abs(result.x) < 0.01
        assert abs(result.z) == pytest.approx(1.0, abs=0.01)


class TestPlatformQueries:
    """Tests for platform query methods."""

    def test_is_on_moving_platform_not_attached(self):
        """is_on_moving_platform should return False when not attached."""
        provider = MockPlatformProvider()
        handler = PlatformHandler(provider)
        assert handler.is_on_moving_platform() is False

    def test_is_on_moving_platform_stationary(self):
        """is_on_moving_platform should return False for stationary platform."""
        provider = MockPlatformProvider()
        provider.platforms[1] = PlatformData(
            platform_id=1,
            velocity=Vector3.zero(),
            angular_velocity=Vector3.zero(),
            is_active=True,
        )
        handler = PlatformHandler(provider)
        handler.attach_to_platform(1, Vector3.zero(), Quaternion.identity())
        assert handler.is_on_moving_platform() is False

    def test_is_on_moving_platform_moving(self):
        """is_on_moving_platform should return True for moving platform."""
        provider = MockPlatformProvider()
        provider.platforms[1] = PlatformData(
            platform_id=1,
            velocity=Vector3(1.0, 0.0, 0.0),
            is_active=True,
        )
        handler = PlatformHandler(provider)
        handler.attach_to_platform(1, Vector3.zero(), Quaternion.identity())
        assert handler.is_on_moving_platform() is True

    def test_is_on_moving_platform_rotating(self):
        """is_on_moving_platform should return True for rotating platform."""
        provider = MockPlatformProvider()
        provider.platforms[1] = PlatformData(
            platform_id=1,
            velocity=Vector3.zero(),
            angular_velocity=Vector3(0.0, 0.1, 0.0),
            is_active=True,
        )
        handler = PlatformHandler(provider)
        handler.attach_to_platform(1, Vector3.zero(), Quaternion.identity())
        assert handler.is_on_moving_platform() is True

    def test_get_platform_type_not_attached(self):
        """get_platform_type should return None when not attached."""
        provider = MockPlatformProvider()
        handler = PlatformHandler(provider)
        assert handler.get_platform_type() is None

    def test_get_platform_type_attached(self):
        """get_platform_type should return platform type."""
        provider = MockPlatformProvider()
        provider.platforms[1] = PlatformData(
            platform_id=1,
            platform_type=PlatformType.ELEVATOR,
        )
        handler = PlatformHandler(provider)
        handler.attach_to_platform(1, Vector3.zero(), Quaternion.identity())
        assert handler.get_platform_type() == PlatformType.ELEVATOR

    def test_get_attachment_duration_not_attached(self):
        """get_attachment_duration should return 0 when not attached."""
        provider = MockPlatformProvider()
        handler = PlatformHandler(provider)
        assert handler.get_attachment_duration(100.0) == 0.0

    def test_get_attachment_duration_attached(self):
        """get_attachment_duration should return time since attachment."""
        provider = MockPlatformProvider()
        provider.platforms[1] = PlatformData(platform_id=1)
        handler = PlatformHandler(provider)
        handler.attach_to_platform(
            1, Vector3.zero(), Quaternion.identity(), time=10.0
        )
        duration = handler.get_attachment_duration(15.0)
        assert duration == 5.0


class TestPlatformDebug:
    """Tests for platform debug info."""

    def test_get_debug_info_not_attached(self):
        """get_debug_info should indicate not attached."""
        provider = MockPlatformProvider()
        handler = PlatformHandler(provider)
        info = handler.get_debug_info()
        assert info["attached"] is False

    def test_get_debug_info_attached(self):
        """get_debug_info should return platform info when attached."""
        provider = MockPlatformProvider()
        provider.platforms[1] = PlatformData(
            platform_id=1,
            platform_type=PlatformType.LINEAR,
        )
        handler = PlatformHandler(provider)
        handler.attach_to_platform(
            1, Vector3(1.0, 0.0, 0.0), Quaternion.identity()
        )
        info = handler.get_debug_info()
        assert info["attached"] is True
        assert info["platform_id"] == 1
        assert info["platform_type"] == "linear"
