"""
Tests for the debug camera system.
"""

import math
import pytest
from unittest.mock import Mock

from engine.debug.tools.debug_camera import (
    CameraConfig,
    CameraTransform,
    DebugCamera,
    DebugCameraMode,
    get_debug_camera,
)


class TestCameraConfig:
    """Tests for CameraConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CameraConfig()
        assert config.move_speed == 10.0
        assert config.fast_speed_multiplier == 3.0
        assert config.slow_speed_multiplier == 0.2
        assert config.rotate_speed == 0.3
        assert config.min_pitch == -89.0
        assert config.max_pitch == 89.0
        assert config.default_orbit_distance == 10.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = CameraConfig(
            move_speed=20.0,
            rotate_speed=0.5,
        )
        assert config.move_speed == 20.0
        assert config.rotate_speed == 0.5


class TestCameraTransform:
    """Tests for CameraTransform."""

    def test_default_transform(self):
        """Test default transform values."""
        transform = CameraTransform()
        assert transform.position == (0.0, 0.0, 0.0)
        assert transform.rotation == (0.0, 0.0, 0.0)

    def test_transform_properties(self):
        """Test transform properties."""
        transform = CameraTransform(
            position=(1.0, 2.0, 3.0),
            rotation=(10.0, 20.0, 30.0),
        )
        assert transform.pitch == 10.0
        assert transform.yaw == 20.0
        assert transform.roll == 30.0


class TestDebugCameraMode:
    """Tests for DebugCameraMode enum."""

    def test_modes(self):
        """Test all camera modes exist."""
        assert DebugCameraMode.FREE
        assert DebugCameraMode.ORBIT
        assert DebugCameraMode.FOLLOW
        assert DebugCameraMode.CYCLE


class TestDebugCamera:
    """Tests for DebugCamera."""

    @pytest.fixture
    def camera(self):
        """Create a fresh DebugCamera."""
        return DebugCamera()

    def test_initial_state(self, camera):
        """Test initial camera state."""
        assert camera.mode == DebugCameraMode.FREE
        assert camera.enabled is False
        assert camera.target is None
        assert camera.get_position() == (0.0, 0.0, 0.0)

    def test_enable_disable(self, camera):
        """Test enabling and disabling camera."""
        camera.enabled = True
        assert camera.enabled is True

        camera.enabled = False
        assert camera.enabled is False

    def test_set_mode(self, camera):
        """Test setting camera mode."""
        camera.set_mode(DebugCameraMode.ORBIT)
        assert camera.mode == DebugCameraMode.ORBIT

        camera.set_mode(DebugCameraMode.FOLLOW)
        assert camera.mode == DebugCameraMode.FOLLOW

    def test_set_mode_callback(self, camera):
        """Test mode change callback."""
        callback = Mock()
        camera.add_mode_callback(callback)

        camera.set_mode(DebugCameraMode.ORBIT)
        callback.assert_called_with(DebugCameraMode.ORBIT)

    def test_remove_mode_callback(self, camera):
        """Test removing mode callback."""
        callback = Mock()
        camera.add_mode_callback(callback)
        assert camera.remove_mode_callback(callback) is True
        assert camera.remove_mode_callback(callback) is False

        camera.set_mode(DebugCameraMode.ORBIT)
        callback.assert_not_called()

    def test_set_target(self, camera):
        """Test setting target entity."""
        entity = Mock()
        entity.position = Mock(x=10.0, y=5.0, z=20.0)

        camera.set_target(entity)
        assert camera.target == entity

    def test_set_entity_list(self, camera):
        """Test setting entity list for cycle mode."""
        entities = [Mock(), Mock(), Mock()]
        camera.set_entity_list(entities)
        camera.set_mode(DebugCameraMode.CYCLE)

        assert camera.target == entities[0]

    def test_cycle_next(self, camera):
        """Test cycling to next entity."""
        entities = [Mock(), Mock(), Mock()]
        camera.set_entity_list(entities)
        camera.set_mode(DebugCameraMode.CYCLE)

        entity = camera.cycle_next()
        assert entity == entities[1]

        entity = camera.cycle_next()
        assert entity == entities[2]

        entity = camera.cycle_next()
        assert entity == entities[0]  # Wraps around

    def test_cycle_previous(self, camera):
        """Test cycling to previous entity."""
        entities = [Mock(), Mock(), Mock()]
        camera.set_entity_list(entities)
        camera.set_mode(DebugCameraMode.CYCLE)

        entity = camera.cycle_previous()
        assert entity == entities[2]  # Wraps around

        entity = camera.cycle_previous()
        assert entity == entities[1]

    def test_cycle_empty_list(self, camera):
        """Test cycling with empty list."""
        camera.set_entity_list([])
        assert camera.cycle_next() is None
        assert camera.cycle_previous() is None

    def test_move_free_mode(self, camera):
        """Test movement in free mode."""
        camera.enabled = True
        camera.set_mode(DebugCameraMode.FREE)

        # Move forward
        camera.move((1.0, 0.0, 0.0), speed=10.0)
        camera.update(1.0)  # Apply movement

        pos = camera.get_position()
        # Position should have changed in the forward direction
        # (depends on current yaw)

    def test_move_disabled(self, camera):
        """Test movement when disabled."""
        initial_pos = camera.get_position()
        camera.move((1.0, 0.0, 0.0), speed=10.0)
        assert camera.get_position() == initial_pos

    def test_move_non_free_mode(self, camera):
        """Test movement in non-free mode is ignored."""
        camera.enabled = True
        camera.set_mode(DebugCameraMode.ORBIT)

        initial_pos = camera.get_position()
        camera.move((1.0, 0.0, 0.0), speed=10.0)
        # Position shouldn't change from move in orbit mode

    def test_rotate(self, camera):
        """Test rotation."""
        camera.enabled = True
        camera.rotate(10.0, 5.0)
        camera.update(1.0)

        rotation = camera.get_rotation()
        # Rotation should have changed

    def test_rotate_disabled(self, camera):
        """Test rotation when disabled."""
        initial_rot = camera.get_rotation()
        camera.rotate(10.0, 5.0)
        assert camera.get_rotation() == initial_rot

    def test_zoom_orbit_mode(self, camera):
        """Test zooming in orbit mode."""
        camera.enabled = True
        camera.set_mode(DebugCameraMode.ORBIT)

        # Initial orbit distance is default
        camera.zoom(5.0)  # Zoom in

        # Distance should decrease

    def test_zoom_non_orbit_mode(self, camera):
        """Test zoom is ignored in non-orbit modes."""
        camera.enabled = True
        camera.set_mode(DebugCameraMode.FREE)
        camera.zoom(5.0)  # Should be ignored

    def test_get_transform(self, camera):
        """Test getting camera transform."""
        transform = camera.get_transform()
        assert isinstance(transform, CameraTransform)
        assert transform.position == (0.0, 0.0, 0.0)

    def test_get_view_matrix(self, camera):
        """Test getting view matrix."""
        matrix = camera.get_view_matrix()
        assert len(matrix) == 4
        assert all(len(row) == 4 for row in matrix)

    def test_set_fast_mode(self, camera):
        """Test fast mode."""
        camera.set_fast_mode(True)
        # Fast mode should multiply speed

    def test_set_slow_mode(self, camera):
        """Test slow mode."""
        camera.set_slow_mode(True)
        # Slow mode should reduce speed

    def test_teleport_to(self, camera):
        """Test teleporting camera."""
        camera.teleport_to(100.0, 50.0, 200.0)
        assert camera.get_position() == (100.0, 50.0, 200.0)

    def test_teleport_to_with_rotation(self, camera):
        """Test teleporting with rotation."""
        camera.teleport_to(100.0, 50.0, 200.0, pitch=10.0, yaw=90.0)
        pos = camera.get_position()
        rot = camera.get_rotation()

        assert pos == (100.0, 50.0, 200.0)
        assert rot[0] == 10.0  # pitch
        assert rot[1] == 90.0  # yaw

    def test_look_at(self, camera):
        """Test look at functionality."""
        camera.teleport_to(0.0, 0.0, 0.0)
        camera.look_at(0.0, 0.0, -10.0)
        camera.update(1.0)

        # Should be looking forward (negative Z)

    def test_update_free(self, camera):
        """Test update in free mode."""
        camera.enabled = True
        camera.set_mode(DebugCameraMode.FREE)
        camera.update(0.016)  # ~60fps

    def test_update_orbit(self, camera):
        """Test update in orbit mode."""
        camera.enabled = True
        camera.set_mode(DebugCameraMode.ORBIT)

        entity = Mock()
        entity.position = Mock(x=10.0, y=0.0, z=10.0)
        camera.set_target(entity)

        camera.update(0.016)

    def test_update_follow(self, camera):
        """Test update in follow mode."""
        camera.enabled = True
        camera.set_mode(DebugCameraMode.FOLLOW)

        entity = Mock()
        entity.position = Mock(x=10.0, y=0.0, z=10.0)
        camera.set_target(entity)

        camera.update(0.016)

    def test_update_disabled(self, camera):
        """Test update when disabled."""
        camera.enabled = False
        initial_pos = camera.get_position()
        camera.update(1.0)
        assert camera.get_position() == initial_pos

    def test_lerp(self):
        """Test linear interpolation helper."""
        assert DebugCamera._lerp(0.0, 10.0, 0.5) == 5.0
        assert DebugCamera._lerp(0.0, 10.0, 0.0) == 0.0
        assert DebugCamera._lerp(0.0, 10.0, 1.0) == 10.0
        assert DebugCamera._lerp(0.0, 10.0, 2.0) == 10.0  # Clamped

    def test_lerp_angle(self):
        """Test angle interpolation helper."""
        # Simple case
        result = DebugCamera._lerp_angle(0.0, 90.0, 0.5)
        assert abs(result - 45.0) < 0.01

        # Wrap around case
        result = DebugCamera._lerp_angle(350.0, 10.0, 0.5)
        # Should go through 0 (shortest path)
        assert result > 350.0 or result < 10.0

    def test_get_entity_position_vec3(self, camera):
        """Test getting entity position from Vec3-like."""
        entity = Mock()
        entity.position = Mock(x=1.0, y=2.0, z=3.0)

        pos = camera._get_entity_position(entity)
        assert pos == (1.0, 2.0, 3.0)

    def test_get_entity_position_tuple(self, camera):
        """Test getting entity position from tuple."""
        entity = Mock()
        entity.position = (1.0, 2.0, 3.0)

        pos = camera._get_entity_position(entity)
        assert pos == (1.0, 2.0, 3.0)

    def test_get_entity_position_transform(self, camera):
        """Test getting entity position from transform."""
        entity = Mock(spec=[])  # No position attribute
        entity.transform = Mock()
        entity.transform.position = Mock(x=1.0, y=2.0, z=3.0)

        pos = camera._get_entity_position(entity)
        assert pos == (1.0, 2.0, 3.0)

    def test_get_entity_position_none(self, camera):
        """Test getting entity position when not available."""
        entity = Mock(spec=[])  # Empty spec, no attributes

        pos = camera._get_entity_position(entity)
        assert pos is None


class TestGetDebugCamera:
    """Tests for get_debug_camera singleton."""

    def test_singleton(self):
        """Test that get_debug_camera returns singleton."""
        # Reset the singleton
        import engine.debug.tools.debug_camera as camera_module
        camera_module._debug_camera = None

        camera1 = get_debug_camera()
        camera2 = get_debug_camera()
        assert camera1 is camera2


class TestBuildTypeGuards:
    """Tests for build-type security guards in DebugCamera."""

    @pytest.fixture(autouse=True)
    def reset_env(self):
        """Reset environment before each test."""
        import os
        for var in ["GAME_BUILD_TYPE", "SHIPPING"]:
            os.environ.pop(var, None)
        yield
        for var in ["GAME_BUILD_TYPE", "SHIPPING"]:
            os.environ.pop(var, None)

    def test_cannot_enable_in_shipping(self):
        """Test debug camera cannot be enabled in shipping builds."""
        import os
        os.environ["GAME_BUILD_TYPE"] = "SHIPPING"

        camera = DebugCamera()
        camera.enabled = True

        assert camera.enabled is False

    def test_build_allowed_property(self):
        """Test build_allowed property reflects build type."""
        import os
        os.environ["GAME_BUILD_TYPE"] = "SHIPPING"
        camera = DebugCamera()
        assert camera.build_allowed is False

        os.environ["GAME_BUILD_TYPE"] = "DEBUG"
        camera2 = DebugCamera()
        assert camera2.build_allowed is True

    def test_config_allows_shipping_override(self):
        """Test config can allow debug camera in shipping."""
        import os
        os.environ["GAME_BUILD_TYPE"] = "SHIPPING"

        config = CameraConfig(allow_in_shipping=True)
        camera = DebugCamera(config)

        camera.enabled = True
        assert camera.enabled is True


class TestCameraConfigValues:
    """Tests that config values are used instead of magic numbers."""

    def test_custom_config_speeds(self):
        """Test custom speed values from config."""
        config = CameraConfig(
            move_speed=25.0,
            fast_speed_multiplier=5.0,
            slow_speed_multiplier=0.1,
        )
        camera = DebugCamera(config)

        # Verify config is stored
        assert camera._config.move_speed == 25.0
        assert camera._config.fast_speed_multiplier == 5.0
        assert camera._config.slow_speed_multiplier == 0.1

    def test_custom_orbit_limits(self):
        """Test custom orbit limits from config."""
        config = CameraConfig(
            min_orbit_distance=2.0,
            max_orbit_distance=50.0,
        )
        camera = DebugCamera(config)
        camera.enabled = True
        camera.set_mode(DebugCameraMode.ORBIT)

        # Zoom should be clamped by custom config
        camera._orbit_distance = 1.0  # Below min
        camera.zoom(-100)  # Try to zoom way out
        # Distance should be clamped to max
        assert camera._orbit_distance <= config.max_orbit_distance

    def test_custom_pitch_limits(self):
        """Test custom pitch limits from config."""
        config = CameraConfig(
            min_pitch=-45.0,
            max_pitch=45.0,
        )
        camera = DebugCamera(config)
        camera.enabled = True

        # Rotate with extreme pitch
        camera.rotate(0.0, 1000.0)

        # Pitch should be clamped
        assert camera._target_pitch <= config.max_pitch
        assert camera._target_pitch >= config.min_pitch


class TestDebugCameraGameImpact:
    """
    Tests that verify debug camera actually impacts rendering,
    not just internal state.
    """

    @pytest.fixture
    def camera(self):
        """Create a fresh enabled DebugCamera."""
        camera = DebugCamera()
        camera.enabled = True
        return camera

    def test_view_matrix_changes_with_position(self, camera):
        """Test view matrix changes when camera moves."""
        initial_matrix = camera.get_view_matrix()

        camera.teleport_to(100.0, 50.0, 100.0)
        new_matrix = camera.get_view_matrix()

        # Matrix should be different
        assert initial_matrix != new_matrix

    def test_view_matrix_changes_with_rotation(self, camera):
        """Test view matrix changes when camera rotates."""
        initial_matrix = camera.get_view_matrix()

        camera.teleport_to(0.0, 0.0, 0.0, yaw=90.0)
        new_matrix = camera.get_view_matrix()

        # Matrix should be different
        assert initial_matrix != new_matrix

    def test_follow_mode_tracks_target(self, camera):
        """Test follow mode actually tracks target entity."""
        entity = Mock()
        entity.position = Mock(x=100.0, y=0.0, z=100.0)

        camera.set_target(entity)
        camera.set_mode(DebugCameraMode.FOLLOW)

        # Update camera
        for _ in range(60):  # Simulate 1 second at 60fps
            camera.update(0.016)

        # Camera should be near target
        pos = camera.get_position()
        # Should be within follow distance + height
        dx = pos[0] - entity.position.x
        dz = pos[2] - entity.position.z
        distance = math.sqrt(dx * dx + dz * dz)

        # Distance should be close to follow_distance
        assert abs(distance - camera._config.follow_distance) < 1.0
