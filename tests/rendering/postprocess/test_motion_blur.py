"""
Tests for Motion Blur System

Tests CameraMotionBlur matrix operations, TileMaxVelocity structure,
ObjectMotionBlur, and MotionBlurEffect integration.
"""

import math
import pytest

from engine.rendering.postprocess.motion_blur import (
    CameraMotion,
    CameraMotionBlur,
    MotionBlurEffect,
    MotionBlurMode,
    MotionBlurQuality,
    MotionBlurSettings,
    ObjectMotionBlur,
    TileMaxVelocity,
)


class TestCameraMotion:
    """Test CameraMotion data class."""

    def test_default_motion(self):
        """Test default camera motion is zero."""
        motion = CameraMotion()

        assert motion.velocity == (0.0, 0.0, 0.0)
        assert motion.angular_velocity == (0.0, 0.0, 0.0)

    def test_calculate_screen_velocity(self):
        """Test screen velocity returns default."""
        motion = CameraMotion()
        dx, dy = motion.calculate_screen_velocity((1.0, 2.0, 3.0))

        assert dx == 0.0
        assert dy == 0.0


class TestCameraMotionBlur:
    """Test CameraMotionBlur calculations."""

    def test_camera_blur_creation(self):
        """Test camera motion blur creation."""
        blur = CameraMotionBlur()
        assert blur is not None

    def test_setup(self):
        """Test camera motion blur setup."""
        blur = CameraMotionBlur()
        blur.setup(1920, 1080)
        # Should not raise

    def test_matrix_multiply_identity(self):
        """Test matrix multiplication with identity."""
        blur = CameraMotionBlur()

        identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

        result = blur._multiply_matrices(identity, identity)

        # Identity * Identity = Identity
        for i in range(4):
            for j in range(4):
                assert abs(result[i][j] - identity[i][j]) < 0.01

    def test_matrix_multiply_known(self):
        """Test matrix multiplication with known inputs."""
        blur = CameraMotionBlur()

        # Simple scale matrix
        scale = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

        # Identity
        identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

        result = blur._multiply_matrices(scale, identity)

        # Scale * Identity = Scale
        assert result[0][0] == 2.0
        assert result[1][1] == 2.0
        assert result[2][2] == 2.0
        assert result[3][3] == 1.0

    def test_matrix_multiply_non_commutative(self):
        """Test matrix multiplication is not commutative."""
        blur = CameraMotionBlur()

        # Translation matrix
        translate = [
            [1.0, 0.0, 0.0, 5.0],
            [0.0, 1.0, 0.0, 3.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

        # Scale matrix
        scale = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

        ab = blur._multiply_matrices(translate, scale)
        ba = blur._multiply_matrices(scale, translate)

        # Matrices should differ
        different = False
        for i in range(4):
            for j in range(4):
                if abs(ab[i][j] - ba[i][j]) > 0.01:
                    different = True
                    break
        assert different

    def test_update_matrices(self):
        """Test update_matrices stores matrices correctly."""
        blur = CameraMotionBlur()

        view = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        proj = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

        blur.update_matrices(view, proj)

        # Current VP should be product
        current = blur._current_view_proj
        assert current is not None


class TestObjectMotionBlur:
    """Test ObjectMotionBlur operations."""

    def test_object_blur_creation(self):
        """Test object motion blur creation."""
        blur = ObjectMotionBlur()
        assert blur is not None

    def test_setup(self):
        """Test object motion blur setup."""
        blur = ObjectMotionBlur()
        blur.setup(1920, 1080, tile_size=16)
        # Should not raise

    def test_get_velocity_buffer(self):
        """Test getting velocity buffer handle."""
        blur = ObjectMotionBlur()
        buffer = blur.get_velocity_buffer()

        assert buffer is None

    def test_process_velocity(self):
        """Test velocity processing."""
        blur = ObjectMotionBlur()
        settings = MotionBlurSettings()

        result = blur.process_velocity(None, settings)
        assert result is None


class TestTileMaxVelocity:
    """Test TileMaxVelocity structure."""

    def test_tile_max_creation(self):
        """Test TileMaxVelocity creation."""
        tile = TileMaxVelocity(tile_size=16)

        assert tile.tile_size == 16

    def test_setup_calculates_tile_counts(self):
        """Test setup calculates correct tile counts."""
        tile = TileMaxVelocity(tile_size=16)
        tile.setup(1920, 1080)

        # 1920 / 16 = 120 tiles horizontally
        # 1080 / 16 = 67.5 -> 68 tiles vertically
        assert tile._tiles_x == 120
        assert tile._tiles_y == 68

    def test_tile_size_setter_clamps(self):
        """Test tile_size setter clamps to valid range."""
        tile = TileMaxVelocity(tile_size=16)

        tile.tile_size = 32
        assert tile.tile_size == 32

        # Below min
        tile.tile_size = 1
        assert tile.tile_size == 4  # Clamped to min

        # Above max
        tile.tile_size = 100
        assert tile.tile_size == 64  # Clamped to max

    def test_tile_dimensions_4k(self):
        """Test tile counts at 4K resolution."""
        tile = TileMaxVelocity(tile_size=16)
        tile.setup(3840, 2160)

        assert tile._tiles_x == 240  # 3840 / 16
        assert tile._tiles_y == 135  # 2160 / 16

    def test_tile_dimensions_small(self):
        """Test tile counts at small resolution."""
        tile = TileMaxVelocity(tile_size=16)
        tile.setup(320, 240)

        assert tile._tiles_x == 20  # 320 / 16
        assert tile._tiles_y == 15  # 240 / 16

    def test_tile_dimensions_custom_tile_size(self):
        """Test tile counts with custom tile size."""
        tile = TileMaxVelocity(tile_size=32)
        tile.setup(1920, 1080)

        assert tile._tiles_x == 60  # 1920 / 32
        assert tile._tiles_y == 34  # 1080 / 32

    def test_non_even_dimensions(self):
        """Test non-even dimensions round up."""
        tile = TileMaxVelocity(tile_size=16)
        tile.setup(100, 100)

        # (100 + 16 - 1) // 16 = 115 // 16 = 7
        assert tile._tiles_x == 7
        assert tile._tiles_y == 7

    def test_get_tile_velocity(self):
        """Test get_tile_velocity returns default."""
        tile = TileMaxVelocity(tile_size=16)
        tile.setup(1920, 1080)

        vx, vy = tile.get_tile_velocity(100, 100)

        assert vx == 0.0
        assert vy == 0.0


class TestMotionBlurSettings:
    """Test MotionBlurSettings dataclass."""

    def test_default_settings(self):
        """Test default motion blur settings."""
        settings = MotionBlurSettings()

        assert settings.mode == MotionBlurMode.COMBINED
        assert settings.quality == MotionBlurQuality.MEDIUM
        assert settings.intensity == 1.0
        assert settings.sample_count == 16
        assert settings.shutter_angle == 180.0

    def test_custom_settings(self):
        """Test custom motion blur settings."""
        settings = MotionBlurSettings(
            mode=MotionBlurMode.CAMERA_ONLY,
            quality=MotionBlurQuality.HIGH,
            intensity=0.7,
        )

        assert settings.mode == MotionBlurMode.CAMERA_ONLY
        assert settings.quality == MotionBlurQuality.HIGH
        assert settings.intensity == 0.7

    def test_settings_lerp(self):
        """Test settings interpolation."""
        settings1 = MotionBlurSettings(intensity=0.5, sample_count=8)
        settings2 = MotionBlurSettings(intensity=1.0, sample_count=16)

        lerped = settings1.lerp(settings2, 0.5)

        assert lerped.intensity == 0.75
        assert lerped.sample_count == 12

    def test_shutter_speed_factor(self):
        """Test shutter speed factor calculation."""
        settings = MotionBlurSettings(shutter_angle=180.0)

        # 180 / 360 = 0.5
        assert settings.shutter_speed_factor == 0.5

        settings = MotionBlurSettings(shutter_angle=90.0)
        assert settings.shutter_speed_factor == 0.25

        settings = MotionBlurSettings(shutter_angle=360.0)
        assert settings.shutter_speed_factor == 1.0

        settings = MotionBlurSettings(shutter_angle=0.0)
        assert settings.shutter_speed_factor == 0.0


class TestMotionBlurEffect:
    """Test MotionBlurEffect integration."""

    def test_effect_creation(self):
        """Test motion blur effect creation."""
        effect = MotionBlurEffect()

        assert effect.name == "MotionBlur"
        assert effect.settings is not None

    def test_effect_with_custom_settings(self):
        """Test effect with custom settings."""
        settings = MotionBlurSettings(
            mode=MotionBlurMode.CAMERA_ONLY,
            intensity=0.5,
        )
        effect = MotionBlurEffect(settings)

        assert effect.settings.mode == MotionBlurMode.CAMERA_ONLY
        assert effect.settings.intensity == 0.5

    def test_effect_required_inputs(self):
        """Test effect required inputs."""
        effect = MotionBlurEffect()
        inputs = effect.get_required_inputs()

        assert "color" in inputs
        assert "depth" in inputs
        assert "velocity" in inputs

    def test_effect_outputs(self):
        """Test effect outputs."""
        effect = MotionBlurEffect()
        outputs = effect.get_outputs()

        assert "color" in outputs

    def test_effect_setup(self):
        """Test effect setup."""
        effect = MotionBlurEffect()
        effect.setup(1920, 1080)
        # Should not raise

    def test_effect_execute_disabled(self):
        """Test effect does nothing when disabled."""
        settings = MotionBlurSettings(enabled=False)
        effect = MotionBlurEffect(settings)

        effect.execute({}, {}, 0.016)
        # Should not raise

    def test_effect_cleanup(self):
        """Test effect cleanup."""
        effect = MotionBlurEffect()
        effect.cleanup()
        # Should not raise

    def test_effect_is_compute(self):
        """Test effect uses compute."""
        effect = MotionBlurEffect()
        assert effect.is_compute_effect() is True

    def test_effect_update_camera(self):
        """Test updating camera matrices."""
        effect = MotionBlurEffect()
        effect.setup(1920, 1080)

        view = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        proj = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

        effect.update_camera(view, proj)
        # Should not raise

    def test_effect_get_velocity_buffer(self):
        """Test getting velocity buffer from effect."""
        effect = MotionBlurEffect()
        buffer = effect.get_velocity_buffer()

        assert buffer is None

    def test_effect_zero_intensity(self):
        """Test effect does nothing with zero intensity."""
        settings = MotionBlurSettings(intensity=0.0)
        effect = MotionBlurEffect(settings)

        effect.execute({}, {}, 0.016)
        # Should not raise

    def test_effect_camera_blur_property(self):
        """Test accessing camera blur processor through effect."""
        effect = MotionBlurEffect()
        assert effect.camera_blur is not None

    def test_effect_object_blur_property(self):
        """Test accessing object blur processor through effect."""
        effect = MotionBlurEffect()
        assert effect.object_blur is not None


class TestMotionBlurMode:
    """Test MotionBlurMode enum."""

    def test_all_modes_exist(self):
        """Test all modes exist."""
        modes = [
            MotionBlurMode.CAMERA_ONLY,
            MotionBlurMode.OBJECT_ONLY,
            MotionBlurMode.COMBINED,
        ]

        for m in modes:
            assert m is not None


class TestMotionBlurQuality:
    """Test MotionBlurQuality enum."""

    def test_all_qualities_exist(self):
        """Test all qualities exist."""
        qualities = [
            MotionBlurQuality.LOW,
            MotionBlurQuality.MEDIUM,
            MotionBlurQuality.HIGH,
            MotionBlurQuality.ULTRA,
        ]

        for q in qualities:
            assert q is not None


class TestMotionBlurNumericalSafety:
    """Test numerical safety in motion blur."""

    def test_matrix_multiply_with_zero(self):
        """Test matrix multiply with zero matrix."""
        blur = CameraMotionBlur()

        zero = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
        identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

        result = blur._multiply_matrices(zero, identity)

        for i in range(4):
            for j in range(4):
                assert result[i][j] == 0.0

    def test_tile_max_negative_dimensions(self):
        """Test tile max setup with very small resolution."""
        tile = TileMaxVelocity(tile_size=16)
        tile.setup(1, 1)

        # Should handle small resolutions gracefully
        assert tile._tiles_x >= 1
        assert tile._tiles_y >= 1
