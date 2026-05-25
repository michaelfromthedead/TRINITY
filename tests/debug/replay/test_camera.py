"""Tests for the replay camera system.

Tests camera modes, position tracking, entity following,
and view matrix generation.
"""

from __future__ import annotations

import math

import pytest

from engine.debug.replay.camera import (
    CameraSettings,
    EntityProvider,
    Mat4,
    ReplayCamera,
    ReplayCameraMode,
    Vec3,
)


# =============================================================================
# Vec3 Tests
# =============================================================================


class TestVec3:
    """Tests for Vec3 helper class."""

    def test_create_vec3(self) -> None:
        """Test creating a Vec3."""
        v = Vec3(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_vec3_default(self) -> None:
        """Test Vec3 default values."""
        v = Vec3()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0

    def test_vec3_add(self) -> None:
        """Test Vec3 addition."""
        a = Vec3(1.0, 2.0, 3.0)
        b = Vec3(4.0, 5.0, 6.0)
        result = a + b
        assert result.x == 5.0
        assert result.y == 7.0
        assert result.z == 9.0

    def test_vec3_sub(self) -> None:
        """Test Vec3 subtraction."""
        a = Vec3(4.0, 5.0, 6.0)
        b = Vec3(1.0, 2.0, 3.0)
        result = a - b
        assert result.x == 3.0
        assert result.y == 3.0
        assert result.z == 3.0

    def test_vec3_mul(self) -> None:
        """Test Vec3 scalar multiplication."""
        v = Vec3(1.0, 2.0, 3.0)
        result = v * 2.0
        assert result.x == 2.0
        assert result.y == 4.0
        assert result.z == 6.0

    def test_vec3_rmul(self) -> None:
        """Test Vec3 reverse scalar multiplication."""
        v = Vec3(1.0, 2.0, 3.0)
        result = 2.0 * v
        assert result.x == 2.0
        assert result.y == 4.0
        assert result.z == 6.0

    def test_vec3_length(self) -> None:
        """Test Vec3 length calculation."""
        v = Vec3(3.0, 4.0, 0.0)
        assert v.length() == 5.0

    def test_vec3_normalized(self) -> None:
        """Test Vec3 normalization."""
        v = Vec3(10.0, 0.0, 0.0)
        n = v.normalized()
        assert abs(n.x - 1.0) < 1e-6
        assert abs(n.y) < 1e-6
        assert abs(n.z) < 1e-6

    def test_vec3_normalized_zero_vector(self) -> None:
        """Test normalizing zero vector returns default."""
        v = Vec3(0.0, 0.0, 0.0)
        n = v.normalized()
        # Should return a valid unit vector (z-forward default)
        assert abs(n.length() - 1.0) < 1e-6

    def test_vec3_dot(self) -> None:
        """Test Vec3 dot product."""
        a = Vec3(1.0, 0.0, 0.0)
        b = Vec3(0.0, 1.0, 0.0)
        assert a.dot(b) == 0.0

        c = Vec3(1.0, 2.0, 3.0)
        d = Vec3(4.0, 5.0, 6.0)
        assert c.dot(d) == 32.0

    def test_vec3_cross(self) -> None:
        """Test Vec3 cross product."""
        x = Vec3(1.0, 0.0, 0.0)
        y = Vec3(0.0, 1.0, 0.0)
        z = x.cross(y)
        assert abs(z.x) < 1e-6
        assert abs(z.y) < 1e-6
        assert abs(z.z - 1.0) < 1e-6

    def test_vec3_lerp(self) -> None:
        """Test Vec3 linear interpolation."""
        a = Vec3(0.0, 0.0, 0.0)
        b = Vec3(10.0, 20.0, 30.0)

        mid = a.lerp(b, 0.5)
        assert abs(mid.x - 5.0) < 1e-6
        assert abs(mid.y - 10.0) < 1e-6
        assert abs(mid.z - 15.0) < 1e-6

    def test_vec3_copy(self) -> None:
        """Test Vec3 copy."""
        v = Vec3(1.0, 2.0, 3.0)
        c = v.copy()
        assert c.x == v.x
        assert c.y == v.y
        assert c.z == v.z
        assert c is not v


# =============================================================================
# Mat4 Tests
# =============================================================================


class TestMat4:
    """Tests for Mat4 helper class."""

    def test_identity(self) -> None:
        """Test identity matrix creation."""
        m = Mat4.identity()
        # Check diagonal is 1, rest is 0
        for i in range(4):
            for j in range(4):
                expected = 1.0 if i == j else 0.0
                actual = m.data[i + j * 4]
                assert abs(actual - expected) < 1e-6

    def test_look_at(self) -> None:
        """Test look_at matrix creation."""
        eye = Vec3(0.0, 0.0, 10.0)
        target = Vec3(0.0, 0.0, 0.0)
        up = Vec3(0.0, 1.0, 0.0)

        m = Mat4.look_at(eye, target, up)

        # Matrix should be valid (16 elements)
        assert len(m.data) == 16


# =============================================================================
# ReplayCameraMode Tests
# =============================================================================


class TestReplayCameraMode:
    """Tests for ReplayCameraMode enum."""

    def test_all_modes_exist(self) -> None:
        """Test that all expected modes exist."""
        assert ReplayCameraMode.FOLLOW
        assert ReplayCameraMode.FREE
        assert ReplayCameraMode.POV
        assert ReplayCameraMode.ORBIT

    def test_modes_are_distinct(self) -> None:
        """Test that modes are distinct values."""
        modes = [
            ReplayCameraMode.FOLLOW,
            ReplayCameraMode.FREE,
            ReplayCameraMode.POV,
            ReplayCameraMode.ORBIT,
        ]
        assert len(set(modes)) == 4


# =============================================================================
# CameraSettings Tests
# =============================================================================


class TestCameraSettings:
    """Tests for CameraSettings dataclass."""

    def test_default_settings(self) -> None:
        """Test default camera settings."""
        settings = CameraSettings()
        assert settings.follow_smooth == 0.1
        assert settings.orbit_distance == 10.0
        assert settings.free_move_speed == 10.0
        assert settings.min_distance == 1.0
        assert settings.max_distance == 100.0

    def test_custom_settings(self) -> None:
        """Test custom camera settings."""
        settings = CameraSettings(
            orbit_distance=20.0,
            free_move_speed=5.0,
        )
        assert settings.orbit_distance == 20.0
        assert settings.free_move_speed == 5.0


# =============================================================================
# Mock Entity Provider
# =============================================================================


class MockEntityProvider:
    """Mock entity provider for testing."""

    def __init__(self) -> None:
        self.entities: dict[int, tuple[Vec3, Vec3]] = {}

    def add_entity(self, entity_id: int, position: Vec3, forward: Vec3) -> None:
        self.entities[entity_id] = (position, forward)

    def get_entity_position(self, entity_id: int) -> Vec3 | None:
        if entity_id in self.entities:
            return self.entities[entity_id][0]
        return None

    def get_entity_forward(self, entity_id: int) -> Vec3 | None:
        if entity_id in self.entities:
            return self.entities[entity_id][1]
        return None


# =============================================================================
# ReplayCamera Tests
# =============================================================================


class TestReplayCamera:
    """Tests for ReplayCamera class."""

    def test_create_camera(self) -> None:
        """Test creating a replay camera."""
        camera = ReplayCamera()
        assert camera.mode == ReplayCameraMode.FREE
        assert camera.target_entity_id is None

    def test_create_camera_with_settings(self) -> None:
        """Test creating camera with custom settings."""
        settings = CameraSettings(orbit_distance=50.0)
        camera = ReplayCamera(settings=settings)
        assert camera.settings.orbit_distance == 50.0

    def test_set_mode(self) -> None:
        """Test setting camera mode."""
        camera = ReplayCamera()

        camera.set_mode(ReplayCameraMode.ORBIT)
        assert camera.mode == ReplayCameraMode.ORBIT

        camera.set_mode(ReplayCameraMode.POV)
        assert camera.mode == ReplayCameraMode.POV

    def test_set_target(self) -> None:
        """Test setting target entity."""
        camera = ReplayCamera()

        camera.set_target(42)
        assert camera.target_entity_id == 42

        camera.set_target(None)
        assert camera.target_entity_id is None

    def test_set_position(self) -> None:
        """Test setting camera position."""
        camera = ReplayCamera()
        camera.set_position(Vec3(10.0, 20.0, 30.0))

        pos = camera.position
        assert pos.x == 10.0
        assert pos.y == 20.0
        assert pos.z == 30.0

    def test_set_look_at(self) -> None:
        """Test setting look target."""
        camera = ReplayCamera()
        camera.set_look_at(Vec3(5.0, 5.0, 5.0))

        target = camera.target_position
        assert target.x == 5.0
        assert target.y == 5.0
        assert target.z == 5.0

    def test_cycle_mode(self) -> None:
        """Test cycling through camera modes."""
        camera = ReplayCamera()
        camera.set_mode(ReplayCameraMode.FOLLOW)

        # Should cycle through all modes
        seen_modes = {camera.mode}
        for _ in range(4):
            camera.cycle_mode()
            seen_modes.add(camera.mode)

        assert len(seen_modes) == 4

    def test_reset(self) -> None:
        """Test resetting camera state."""
        camera = ReplayCamera()
        camera.set_position(Vec3(100.0, 100.0, 100.0))
        camera.set_mode(ReplayCameraMode.ORBIT)

        camera.reset()

        pos = camera.position
        assert pos.x == 0.0
        assert pos.y == 5.0
        assert pos.z == -10.0


# =============================================================================
# Camera Mode Update Tests
# =============================================================================


class TestReplayCameraUpdate:
    """Tests for camera update behavior."""

    def get_camera_with_entity(self) -> tuple[ReplayCamera, MockEntityProvider]:
        """Create a camera with a mock entity."""
        provider = MockEntityProvider()
        provider.add_entity(
            1,
            Vec3(10.0, 0.0, 10.0),
            Vec3(0.0, 0.0, 1.0),
        )

        camera = ReplayCamera(entity_provider=provider)
        camera.set_target(1)
        return camera, provider

    def test_update_follow_mode(self) -> None:
        """Test update in FOLLOW mode."""
        camera, provider = self.get_camera_with_entity()
        camera.set_mode(ReplayCameraMode.FOLLOW)

        # Multiple updates to let camera catch up
        for _ in range(100):
            camera.update(1 / 60)

        # Camera should be near the entity
        pos = camera.position
        target = camera.target_position

        # Target should be entity position
        assert abs(target.x - 10.0) < 1e-6
        assert abs(target.z - 10.0) < 1e-6

    def test_update_orbit_mode(self) -> None:
        """Test update in ORBIT mode."""
        camera, provider = self.get_camera_with_entity()
        camera.set_mode(ReplayCameraMode.ORBIT)
        camera.set_orbit_distance(10.0)
        camera.set_orbit_angles(0.0, 0.0)

        camera.update(1 / 60)

        # Camera should be at orbit distance from target
        pos = camera.position
        target = camera.target_position
        diff = pos - target
        distance = diff.length()
        assert abs(distance - 10.0) < 1e-6

    def test_update_pov_mode(self) -> None:
        """Test update in POV mode."""
        camera, provider = self.get_camera_with_entity()
        camera.set_mode(ReplayCameraMode.POV)

        camera.update(1 / 60)

        # Camera should be at entity position + eye offset
        pos = camera.position
        entity_pos = provider.get_entity_position(1)
        assert entity_pos is not None

        # Should be at entity Y + eye height
        assert pos.x == entity_pos.x
        assert pos.z == entity_pos.z

    def test_update_free_mode(self) -> None:
        """Test update in FREE mode with input."""
        camera = ReplayCamera()
        camera.set_mode(ReplayCameraMode.FREE)
        camera.set_position(Vec3(0.0, 0.0, 0.0))

        # Set move input (forward)
        camera.set_free_input(move=Vec3(0.0, 0.0, 1.0))

        initial_z = camera.position.z
        for _ in range(10):
            camera.update(1 / 60)

        # Camera should have moved forward
        final_z = camera.position.z
        assert final_z > initial_z


# =============================================================================
# Camera Orbit Controls Tests
# =============================================================================


class TestReplayCameraOrbit:
    """Tests for orbit camera controls."""

    def test_set_orbit_distance(self) -> None:
        """Test setting orbit distance."""
        camera = ReplayCamera()
        camera.set_orbit_distance(25.0)
        assert camera.settings.orbit_distance == 25.0

    def test_set_orbit_distance_clamps(self) -> None:
        """Test orbit distance clamping."""
        settings = CameraSettings(min_distance=5.0, max_distance=50.0)
        camera = ReplayCamera(settings=settings)

        camera.set_orbit_distance(2.0)
        assert camera.settings.orbit_distance == 5.0

        camera.set_orbit_distance(100.0)
        assert camera.settings.orbit_distance == 50.0

    def test_set_orbit_angles(self) -> None:
        """Test setting orbit angles."""
        camera = ReplayCamera()
        camera.set_orbit_angles(math.pi / 4, math.pi / 6)

        # Angles should be set (we can verify via update behavior)
        # Just check it doesn't raise
        camera.update(1 / 60)

    def test_set_orbit_angles_clamps_pitch(self) -> None:
        """Test that pitch is clamped to avoid gimbal lock."""
        camera = ReplayCamera()

        # Try to set extreme pitch
        camera.set_orbit_angles(0.0, math.pi / 2 + 0.5)

        # Just verify it doesn't crash and update works
        camera.update(1 / 60)


# =============================================================================
# Camera Zoom Tests
# =============================================================================


class TestReplayCameraZoom:
    """Tests for camera zoom functionality."""

    def test_zoom_in_orbit_mode(self) -> None:
        """Test zoom in orbit mode."""
        settings = CameraSettings(orbit_distance=20.0)
        camera = ReplayCamera(settings=settings)
        camera.set_mode(ReplayCameraMode.ORBIT)

        camera.zoom(-5.0)  # Negative = zoom in
        assert camera.settings.orbit_distance == 15.0

        camera.zoom(10.0)  # Positive = zoom out
        assert camera.settings.orbit_distance == 25.0

    def test_zoom_respects_limits(self) -> None:
        """Test zoom respects min/max limits."""
        settings = CameraSettings(
            orbit_distance=20.0,
            min_distance=10.0,
            max_distance=30.0,
        )
        camera = ReplayCamera(settings=settings)
        camera.set_mode(ReplayCameraMode.ORBIT)

        camera.zoom(-100.0)  # Try to zoom way in
        assert camera.settings.orbit_distance == 10.0

        camera.zoom(100.0)  # Try to zoom way out
        assert camera.settings.orbit_distance == 30.0


# =============================================================================
# View Matrix Tests
# =============================================================================


class TestReplayCameraViewMatrix:
    """Tests for view matrix generation."""

    def test_get_view_matrix(self) -> None:
        """Test getting view matrix."""
        camera = ReplayCamera()
        camera.set_position(Vec3(0.0, 5.0, -10.0))
        camera.set_look_at(Vec3(0.0, 0.0, 0.0))

        matrix = camera.get_view_matrix()

        assert isinstance(matrix, Mat4)
        assert len(matrix.data) == 16

    def test_view_matrix_changes_with_position(self) -> None:
        """Test that view matrix changes when camera moves."""
        camera = ReplayCamera()

        camera.set_position(Vec3(0.0, 0.0, 10.0))
        camera.set_look_at(Vec3(0.0, 0.0, 0.0))
        m1 = camera.get_view_matrix()

        camera.set_position(Vec3(10.0, 0.0, 0.0))
        m2 = camera.get_view_matrix()

        # Matrices should be different
        assert m1.data != m2.data


# =============================================================================
# Entity Provider Protocol Tests
# =============================================================================


class TestEntityProviderProtocol:
    """Tests for EntityProvider protocol."""

    def test_mock_provider_implements_protocol(self) -> None:
        """Test that mock provider implements the protocol."""
        provider = MockEntityProvider()
        assert isinstance(provider, EntityProvider)

    def test_provider_returns_none_for_missing(self) -> None:
        """Test that provider returns None for missing entity."""
        provider = MockEntityProvider()
        assert provider.get_entity_position(999) is None
        assert provider.get_entity_forward(999) is None

    def test_camera_handles_missing_entity(self) -> None:
        """Test that camera handles missing entity gracefully."""
        provider = MockEntityProvider()
        camera = ReplayCamera(entity_provider=provider)
        camera.set_target(999)  # Non-existent entity
        camera.set_mode(ReplayCameraMode.FOLLOW)

        # Should not crash
        camera.update(1 / 60)
