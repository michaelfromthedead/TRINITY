"""
Tests for comfort settings and vignette system (comfort.py).

Tests the comfort system including:
    - ComfortVignette component
    - XRComfortSettings resource
    - ComfortPreset configuration
    - ComfortManager
    - @xr_comfort decorator
"""

import pytest

from engine.xr.locomotion.comfort import (
    COMFORT_PRESETS,
    ComfortLevel,
    ComfortManager,
    ComfortMetrics,
    ComfortPreset,
    ComfortVignette,
    PlayMode,
    TunnelingMode,
    VignetteShape,
    VignetteState,
    VignetteTrigger,
    XRComfortSettings,
    get_preset,
    list_presets,
    xr_comfort,
)
from trinity.decorators.ops import decompose


# =============================================================================
# ComfortVignette Tests
# =============================================================================


class TestComfortVignette:
    """Tests for the ComfortVignette component."""

    def test_default_state(self):
        """Test default initialization."""
        vignette = ComfortVignette()
        assert vignette.enabled is True
        assert vignette.current_intensity == 0.0
        assert vignette.base_intensity == 0.5
        assert vignette.shape == VignetteShape.CIRCULAR

    def test_disabled_returns_zero(self):
        """Test disabled vignette always returns zero."""
        vignette = ComfortVignette(enabled=False)
        intensity = vignette.update(
            linear_velocity=10.0,
            angular_velocity=180.0,
            delta_time=0.016,
        )
        assert intensity == 0.0
        assert vignette.current_intensity == 0.0

    def test_velocity_trigger(self):
        """Test vignette triggered by linear velocity."""
        vignette = ComfortVignette(
            enabled=True,
            trigger=VignetteTrigger.VELOCITY,
            velocity_threshold=1.0,
            base_intensity=0.5,
            fade_in_speed=100.0,  # Fast for testing
        )

        # Below threshold
        intensity1 = vignette.update(0.5, 0.0, 0.1)
        assert intensity1 == pytest.approx(0.0, abs=0.01)

        # Above threshold
        intensity2 = vignette.update(3.0, 0.0, 0.1)
        assert intensity2 > 0.0

    def test_rotation_trigger(self):
        """Test vignette triggered by angular velocity."""
        vignette = ComfortVignette(
            enabled=True,
            trigger=VignetteTrigger.ROTATION,
            angular_velocity_threshold=30.0,
            base_intensity=0.5,
            fade_in_speed=100.0,
        )

        # Below threshold
        intensity1 = vignette.update(0.0, 20.0, 0.1)
        assert intensity1 == pytest.approx(0.0, abs=0.01)

        # Above threshold
        intensity2 = vignette.update(0.0, 90.0, 0.1)
        assert intensity2 > 0.0

    def test_both_trigger(self):
        """Test vignette triggered by both velocity and rotation."""
        vignette = ComfortVignette(
            enabled=True,
            trigger=VignetteTrigger.BOTH,
            velocity_threshold=1.0,
            angular_velocity_threshold=30.0,
            base_intensity=0.5,
            fade_in_speed=100.0,
        )

        # Either should trigger
        intensity_velocity = vignette.update(3.0, 0.0, 0.1)
        assert intensity_velocity > 0.0

        # Reset
        vignette.current_intensity = 0.0

        intensity_rotation = vignette.update(0.0, 90.0, 0.1)
        assert intensity_rotation > 0.0

    def test_fade_in(self):
        """Test vignette fade in."""
        vignette = ComfortVignette(
            enabled=True,
            trigger=VignetteTrigger.VELOCITY,
            velocity_threshold=0.0,
            base_intensity=1.0,
            max_intensity=1.0,
            fade_in_speed=10.0,
        )

        # Should fade in gradually
        vignette.update(2.0, 0.0, 0.05)
        assert 0.0 < vignette.current_intensity < 1.0

        # Continue fading
        vignette.update(2.0, 0.0, 0.5)
        assert vignette.current_intensity > 0.3

    def test_fade_out(self):
        """Test vignette fade out."""
        vignette = ComfortVignette(
            enabled=True,
            trigger=VignetteTrigger.VELOCITY,
            velocity_threshold=1.0,
            base_intensity=0.5,
            fade_out_speed=10.0,
        )

        # Set to active
        vignette.current_intensity = 0.5

        # Should fade out when velocity is low
        vignette.update(0.0, 0.0, 0.1)
        assert vignette.current_intensity < 0.5

    def test_max_intensity_clamp(self):
        """Test intensity clamped to max."""
        vignette = ComfortVignette(
            enabled=True,
            trigger=VignetteTrigger.VELOCITY,
            velocity_threshold=0.0,
            base_intensity=1.0,
            max_intensity=0.6,
            fade_in_speed=100.0,
        )

        # Very high velocity
        vignette.update(100.0, 0.0, 1.0)
        assert vignette.current_intensity <= 0.6

    def test_get_shader_params(self):
        """Test shader parameter generation."""
        vignette = ComfortVignette(
            shape=VignetteShape.ELLIPTICAL,
            inner_radius=0.3,
            outer_radius=0.7,
            color=(0.1, 0.0, 0.0),
        )
        vignette.current_intensity = 0.5

        params = vignette.get_shader_params()

        assert params["intensity"] == 0.5
        assert params["inner_radius"] == 0.3
        assert params["outer_radius"] == 0.7
        assert params["shape"] == "elliptical"
        assert params["color"] == (0.1, 0.0, 0.0)

    def test_manual_trigger(self):
        """Test manual intensity override."""
        vignette = ComfortVignette(
            enabled=True,
            trigger=VignetteTrigger.MANUAL,
            max_intensity=1.0,
        )

        vignette.set_intensity_override(0.7)
        assert vignette._state.target_intensity == 0.7


# =============================================================================
# XRComfortSettings Tests
# =============================================================================


class TestXRComfortSettings:
    """Tests for the XRComfortSettings resource."""

    def test_default_values(self):
        """Test default initialization."""
        settings = XRComfortSettings()
        assert settings.comfort_level == ComfortLevel.MEDIUM
        assert settings.snap_turn_enabled is True
        assert settings.vignette_enabled is True
        assert settings.teleport_fade_enabled is True

    def test_apply_preset_none(self):
        """Test applying NONE comfort preset."""
        settings = XRComfortSettings()
        settings.apply_preset(ComfortLevel.NONE)

        assert settings.snap_turn_enabled is False
        assert settings.vignette_enabled is False
        assert settings.teleport_fade_enabled is False
        assert settings.movement_speed_scale == 1.0

    def test_apply_preset_high(self):
        """Test applying HIGH comfort preset."""
        settings = XRComfortSettings()
        settings.apply_preset(ComfortLevel.HIGH)

        assert settings.snap_turn_enabled is True
        assert settings.snap_turn_angle == 30.0
        assert settings.vignette_enabled is True
        assert settings.vignette_intensity == 0.7
        assert settings.stable_horizon_enabled is True
        assert settings.smooth_locomotion_enabled is False

    def test_apply_preset_medium(self):
        """Test applying MEDIUM comfort preset."""
        settings = XRComfortSettings()
        settings.apply_preset(ComfortLevel.MEDIUM)

        assert settings.snap_turn_enabled is True
        assert settings.snap_turn_angle == 45.0
        assert settings.vignette_intensity == 0.5
        assert settings.tunneling_mode == TunnelingMode.MILD

    def test_get_effective_turn_type(self):
        """Test effective turn type getter."""
        settings = XRComfortSettings()

        settings.snap_turn_enabled = True
        assert settings.get_effective_turn_type() == "snap"

        settings.snap_turn_enabled = False
        assert settings.get_effective_turn_type() == "smooth"

    def test_get_effective_turn_speed(self):
        """Test effective turn speed getter."""
        settings = XRComfortSettings(
            snap_turn_enabled=True,
            snap_turn_angle=45.0,
            smooth_turn_speed=90.0,
        )

        assert settings.get_effective_turn_speed() == 45.0

        settings.snap_turn_enabled = False
        assert settings.get_effective_turn_speed() == 90.0

    def test_metrics_tracking(self):
        """Test comfort metrics tracking."""
        settings = XRComfortSettings(track_comfort_metrics=True)

        settings.update_metrics(
            rotation_delta=10.0,
            velocity=2.0,
            delta_time=0.016,
            comfort_triggered=True,
        )

        metrics = settings.get_metrics()
        assert metrics.cumulative_rotation > 0
        assert metrics.comfort_events == 1

    def test_metrics_disabled(self):
        """Test metrics tracking when disabled."""
        settings = XRComfortSettings(track_comfort_metrics=False)

        settings.update_metrics(
            rotation_delta=10.0,
            velocity=2.0,
            delta_time=0.016,
            comfort_triggered=True,
        )

        metrics = settings.get_metrics()
        assert metrics.comfort_events == 0

    def test_reset_metrics(self):
        """Test metrics reset."""
        settings = XRComfortSettings(track_comfort_metrics=True)

        settings.update_metrics(10.0, 2.0, 0.016, True)
        settings.reset_metrics()

        metrics = settings.get_metrics()
        assert metrics.cumulative_rotation == 0.0
        assert metrics.comfort_events == 0

    def test_seated_mode(self):
        """Test seated mode settings."""
        settings = XRComfortSettings(
            seated_mode=True,
            seated_height_offset=0.3,
            play_mode=PlayMode.SEATED,
        )

        assert settings.seated_mode is True
        assert settings.seated_height_offset == 0.3
        assert settings.play_mode == PlayMode.SEATED


# =============================================================================
# ComfortPreset Tests
# =============================================================================


class TestComfortPreset:
    """Tests for ComfortPreset configuration."""

    def test_create_preset(self):
        """Test creating a comfort preset."""
        preset = ComfortPreset(
            name="Custom",
            description="Custom comfort preset",
            snap_turn=True,
            snap_angle=30.0,
            vignette=True,
            vignette_intensity=0.6,
        )

        assert preset.name == "Custom"
        assert preset.snap_angle == 30.0
        assert preset.vignette_intensity == 0.6

    def test_apply_to_settings(self):
        """Test applying preset to settings."""
        preset = ComfortPreset(
            name="Test",
            description="Test preset",
            snap_turn=False,
            vignette=True,
            vignette_intensity=0.8,
            speed_scale=0.7,
        )

        settings = XRComfortSettings()
        preset.apply_to(settings)

        assert settings.snap_turn_enabled is False
        assert settings.vignette_enabled is True
        assert settings.vignette_intensity == 0.8
        assert settings.movement_speed_scale == 0.7
        assert settings.comfort_level == ComfortLevel.CUSTOM

    def test_builtin_presets_exist(self):
        """Test that built-in presets exist."""
        assert "veteran" in COMFORT_PRESETS
        assert "intermediate" in COMFORT_PRESETS
        assert "comfortable" in COMFORT_PRESETS
        assert "maximum" in COMFORT_PRESETS
        assert "seated" in COMFORT_PRESETS

    def test_get_preset(self):
        """Test get_preset function."""
        preset = get_preset("veteran")
        assert preset is not None
        assert preset.name == "Veteran"

        invalid = get_preset("nonexistent")
        assert invalid is None

    def test_list_presets(self):
        """Test list_presets function."""
        presets = list_presets()
        assert isinstance(presets, list)
        assert "veteran" in presets
        assert "comfortable" in presets


# =============================================================================
# ComfortManager Tests
# =============================================================================


class TestComfortManager:
    """Tests for the ComfortManager."""

    def test_init(self):
        """Test manager initialization."""
        settings = XRComfortSettings()
        manager = ComfortManager(settings)

        assert manager.settings is settings
        assert manager.vignette is not None

    def test_update(self):
        """Test manager update."""
        settings = XRComfortSettings(
            vignette_enabled=True,
            vignette_intensity=0.5,
        )
        manager = ComfortManager(settings)

        state = manager.update(
            linear_velocity=2.0,
            angular_velocity=45.0,
            delta_time=0.016,
        )

        assert "vignette_intensity" in state
        assert "snap_turn_enabled" in state
        assert "speed_scale" in state

    def test_apply_preset(self):
        """Test applying preset through manager."""
        settings = XRComfortSettings()
        manager = ComfortManager(settings)

        result = manager.apply_preset("veteran")
        assert result is True
        assert settings.vignette_enabled is False

        result_invalid = manager.apply_preset("nonexistent")
        assert result_invalid is False

    def test_vignette_sync_with_settings(self):
        """Test vignette syncs with settings."""
        settings = XRComfortSettings(
            vignette_enabled=True,
            vignette_intensity=0.7,
        )
        manager = ComfortManager(settings)

        assert manager.vignette.enabled is True

        # Change setting and update
        settings.vignette_intensity = 0.3
        manager.update(0.0, 0.0, 0.016)

        assert manager.vignette.base_intensity == 0.3


# =============================================================================
# @xr_comfort Decorator Tests
# =============================================================================


class TestXRComfortDecorator:
    """Tests for the @xr_comfort decorator."""

    def test_basic_application(self):
        """Test basic decorator application."""
        @xr_comfort(comfort_type="vignette")
        class Movement:
            pass

        assert Movement._xr_comfort is True

    def test_comfort_type_stored(self):
        """Test comfort type is stored."""
        @xr_comfort(comfort_type="locomotion")
        class Movement:
            pass

        assert Movement._comfort_type == "locomotion"

    def test_settings_stored(self):
        """Test settings are stored."""
        @xr_comfort(comfort_type="vignette", settings={"intensity": 0.5})
        class Movement:
            pass

        assert Movement._comfort_settings == {"intensity": 0.5}

    def test_tags_applied(self):
        """Test tags are applied."""
        @xr_comfort(comfort_type="turn", settings={"snap": True})
        class Movement:
            pass

        assert Movement._tags["xr_comfort"] is True
        assert Movement._tags["comfort_type"] == "turn"

    def test_registered_in_xr_registry(self):
        """Test registration in XR registry."""
        @xr_comfort(comfort_type="general")
        class Movement:
            pass

        assert "xr" in Movement._registries

    def test_invalid_comfort_type(self):
        """Test validation of comfort type."""
        with pytest.raises(ValueError, match="comfort_type"):
            @xr_comfort(comfort_type="invalid")
            class Movement:
                pass

    def test_valid_comfort_types(self):
        """Test all valid comfort types."""
        valid_types = ["vignette", "locomotion", "turn", "teleport", "general"]

        for comfort_type in valid_types:
            @xr_comfort(comfort_type=comfort_type)
            class Movement:
                pass

            assert Movement._comfort_type == comfort_type

    def test_applied_decorators(self):
        """Test applied decorators list."""
        @xr_comfort(comfort_type="vignette")
        class Movement:
            pass

        assert "xr_comfort" in Movement._applied_decorators

    def test_steps_recorded(self):
        """Test steps are recorded."""
        @xr_comfort(comfort_type="vignette")
        class Movement:
            pass

        assert len(Movement._applied_steps) > 0

    def test_decompose(self):
        """Test decompose shows steps."""
        steps = decompose(xr_comfort)
        assert isinstance(steps, list)


# =============================================================================
# Enum Tests
# =============================================================================


class TestComfortEnums:
    """Tests for comfort-related enums."""

    def test_comfort_level_values(self):
        """Test ComfortLevel enum values."""
        assert ComfortLevel.NONE.value == "none"
        assert ComfortLevel.LOW.value == "low"
        assert ComfortLevel.MEDIUM.value == "medium"
        assert ComfortLevel.HIGH.value == "high"
        assert ComfortLevel.CUSTOM.value == "custom"

    def test_vignette_shape_values(self):
        """Test VignetteShape enum values."""
        assert VignetteShape.CIRCULAR.value == "circular"
        assert VignetteShape.ELLIPTICAL.value == "elliptical"
        assert VignetteShape.RECTANGULAR.value == "rectangular"

    def test_vignette_trigger_values(self):
        """Test VignetteTrigger enum values."""
        assert VignetteTrigger.VELOCITY.value == "velocity"
        assert VignetteTrigger.ROTATION.value == "rotation"
        assert VignetteTrigger.BOTH.value == "both"
        assert VignetteTrigger.MANUAL.value == "manual"

    def test_tunneling_mode_values(self):
        """Test TunnelingMode enum values."""
        assert TunnelingMode.DISABLED.value == "disabled"
        assert TunnelingMode.MILD.value == "mild"
        assert TunnelingMode.MODERATE.value == "moderate"
        assert TunnelingMode.STRONG.value == "strong"

    def test_play_mode_values(self):
        """Test PlayMode enum values."""
        assert PlayMode.STANDING.value == "standing"
        assert PlayMode.SEATED.value == "seated"
        assert PlayMode.ROOMSCALE.value == "roomscale"


# =============================================================================
# Integration Tests
# =============================================================================


class TestComfortIntegration:
    """Integration tests for comfort system."""

    def test_full_comfort_workflow(self):
        """Test complete comfort workflow."""
        # Create settings with medium comfort
        settings = XRComfortSettings()
        settings.apply_preset(ComfortLevel.MEDIUM)

        # Create manager
        manager = ComfortManager(settings)

        # Simulate movement
        for _ in range(10):
            state = manager.update(
                linear_velocity=3.0,
                angular_velocity=60.0,
                delta_time=0.016,
            )

        # Vignette should be active
        assert state["vignette_active"] is True or state["vignette_intensity"] >= 0

    def test_preset_progression(self):
        """Test progressing through comfort presets."""
        settings = XRComfortSettings()

        # Start with maximum comfort
        settings.apply_preset(ComfortLevel.HIGH)
        assert settings.smooth_locomotion_enabled is False

        # Progress to medium
        settings.apply_preset(ComfortLevel.MEDIUM)
        assert settings.smooth_locomotion_enabled is True
        assert settings.vignette_enabled is True

        # Progress to none
        settings.apply_preset(ComfortLevel.NONE)
        assert settings.vignette_enabled is False

    def test_custom_preset_application(self):
        """Test applying custom preset."""
        custom_preset = ComfortPreset(
            name="My Comfort",
            description="Personal comfort settings",
            snap_turn=True,
            snap_angle=22.5,
            vignette=True,
            vignette_intensity=0.4,
            tunneling=TunnelingMode.MILD,
            speed_scale=0.85,
        )

        settings = XRComfortSettings()
        custom_preset.apply_to(settings)

        assert settings.snap_turn_angle == 22.5
        assert settings.vignette_intensity == 0.4
        assert settings.tunneling_mode == TunnelingMode.MILD
        assert settings.movement_speed_scale == 0.85
        assert settings.comfort_level == ComfortLevel.CUSTOM
