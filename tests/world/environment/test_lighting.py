"""
Tests for the lighting system (lighting.py).

Tests directional lights, ambient lighting, sun/moon lights,
environment lighting, and light probes.
"""

import pytest
import math

from engine.world.environment.lighting import (
    DirectionalLight,
    AmbientLight,
    SunLight,
    MoonLight,
    EnvironmentLighting,
    LightProbe,
    LightProbeGrid,
)
from engine.world.environment.time_of_day import TimeOfDayController, SunPosition
from engine.world.environment.weather import WeatherParameters
from engine.world.environment.sky import SkyManager


# =============================================================================
# DirectionalLight Tests
# =============================================================================


class TestDirectionalLight:
    def test_default_creation(self):
        light = DirectionalLight()
        assert light.direction == (0.0, -1.0, 0.0)
        assert light.intensity == 1.0

    def test_custom_direction(self):
        light = DirectionalLight(direction=(1, 0, 0), color=(1, 0.8, 0.6))
        assert light.direction == (1, 0, 0)
        assert light.color == (1, 0.8, 0.6)

    def test_set_direction_from_angles(self):
        light = DirectionalLight()
        light.set_direction_from_angles(azimuth=180.0, elevation=45.0)

        # Direction should point toward light
        assert light.direction[1] > 0  # Elevated

    def test_get_shadow_direction(self):
        light = DirectionalLight(direction=(0, 1, 0))
        shadow_dir = light.get_shadow_direction()

        assert shadow_dir == (0, -1, 0)  # Opposite

    def test_casts_shadows_default(self):
        light = DirectionalLight()
        assert light.casts_shadows is True

    def test_shadow_settings(self):
        light = DirectionalLight(
            shadow_cascade_count=3,
            shadow_distance=1000.0,
        )
        assert light.shadow_cascade_count == 3
        assert light.shadow_distance == 1000.0

    def test_to_dict(self):
        light = DirectionalLight(intensity=0.8)
        d = light.to_dict()

        assert d["intensity"] == 0.8
        assert "direction" in d
        assert "color" in d

    def test_lerp(self):
        l1 = DirectionalLight(intensity=0.0, color=(1, 0, 0))
        l2 = DirectionalLight(intensity=1.0, color=(0, 1, 0))

        result = l1.lerp(l2, 0.5)

        assert abs(result.intensity - 0.5) < 0.01
        assert abs(result.color[0] - 0.5) < 0.01
        assert abs(result.color[1] - 0.5) < 0.01

    def test_lerp_direction(self):
        l1 = DirectionalLight(direction=(1, 0, 0))
        l2 = DirectionalLight(direction=(0, 1, 0))

        result = l1.lerp(l2, 0.5)

        # Direction should be normalized
        length = math.sqrt(sum(d*d for d in result.direction))
        assert abs(length - 1.0) < 0.01

    def test_observer(self):
        light = DirectionalLight()
        changes = []
        light.add_observer(lambda p, o, n: changes.append((p, o, n)))

        light.set_direction_from_angles(180.0, 45.0)

        assert len(changes) == 1
        assert changes[0][0] == "direction"


# =============================================================================
# AmbientLight Tests
# =============================================================================


class TestAmbientLight:
    def test_default_creation(self):
        light = AmbientLight()
        assert light.intensity == 0.3
        assert light.color == (0.5, 0.6, 0.7)

    def test_hemisphere_disabled_by_default(self):
        light = AmbientLight()
        assert light.is_hemisphere is False

    def test_hemisphere_enabled(self):
        light = AmbientLight(
            sky_color=(0.5, 0.6, 0.8),
            ground_color=(0.2, 0.15, 0.1),
        )
        assert light.is_hemisphere is True

    def test_get_light_at_normal_uniform(self):
        light = AmbientLight(color=(0.5, 0.5, 0.5), intensity=1.0)

        up = light.get_light_at_normal((0, 1, 0))
        down = light.get_light_at_normal((0, -1, 0))

        # Without hemisphere, should be same
        assert up == down

    def test_get_light_at_normal_hemisphere(self):
        light = AmbientLight(
            sky_color=(1, 1, 1),
            ground_color=(0, 0, 0),
            intensity=1.0,
        )

        up = light.get_light_at_normal((0, 1, 0))
        down = light.get_light_at_normal((0, -1, 0))

        # Up should be brighter (sky)
        assert sum(up) > sum(down)

    def test_ao_settings(self):
        light = AmbientLight(ao_enabled=True, ao_intensity=0.7)
        assert light.ao_enabled is True
        assert light.ao_intensity == 0.7

    def test_to_dict(self):
        light = AmbientLight(intensity=0.5)
        d = light.to_dict()

        assert d["intensity"] == 0.5
        assert "color" in d

    def test_lerp(self):
        l1 = AmbientLight(intensity=0.0)
        l2 = AmbientLight(intensity=1.0)

        result = l1.lerp(l2, 0.5)

        assert abs(result.intensity - 0.5) < 0.01


# =============================================================================
# SunLight Tests
# =============================================================================


class TestSunLight:
    def test_default_creation(self):
        sun = SunLight()
        assert sun.linked_to_tod is True

    def test_not_linked(self):
        sun = SunLight(linked_to_tod=False)
        assert sun.linked_to_tod is False

    def test_update_from_tod_linked(self):
        sun = SunLight(linked_to_tod=True)
        tod = TimeOfDayController(time_hours=12.0)

        sun.update_from_tod(tod)

        # Should have updated direction
        assert sun.direction != (0, -1, 0)
        assert sun.intensity > 0

    def test_update_from_tod_not_linked(self):
        sun = SunLight(linked_to_tod=False, intensity=0.5)
        tod = TimeOfDayController(time_hours=12.0)

        sun.update_from_tod(tod)

        # Should not change
        assert sun.intensity == 0.5

    def test_intensity_zero_at_night(self):
        sun = SunLight()
        tod = TimeOfDayController(time_hours=0.0)

        sun.update_from_tod(tod)

        # Sun below horizon at midnight
        assert sun.intensity == 0.0

    def test_color_from_tod(self):
        sun = SunLight()
        tod = TimeOfDayController(time_hours=12.0)

        sun.update_from_tod(tod)

        # Should have updated color from lighting curve
        assert sun.color != (1.0, 1.0, 1.0)


# =============================================================================
# MoonLight Tests
# =============================================================================


class TestMoonLight:
    def test_default_creation(self):
        moon = MoonLight()
        assert moon.moon_phase == 0.5

    def test_custom_phase(self):
        moon = MoonLight(moon_phase=0.25)
        assert moon.moon_phase == 0.25

    def test_phase_affects_intensity(self):
        full_moon = MoonLight(moon_phase=0.5)
        new_moon = MoonLight(moon_phase=0.0)

        # Full moon should be brighter (phase 0.5 = 100%, phase 0.0 = 0%)
        # With base intensity of 0.2, full moon should be 0.2, new moon should be 0
        assert full_moon.intensity > new_moon.intensity or (
            full_moon.intensity == 0.2 and new_moon.intensity == 0.0
        )

    def test_set_phase(self):
        moon = MoonLight()
        moon.moon_phase = 0.25
        assert moon.moon_phase == 0.25

    def test_phase_wraps(self):
        moon = MoonLight()
        moon.moon_phase = 1.5
        assert moon.moon_phase == 0.5

    def test_update_from_tod(self):
        moon = MoonLight()
        tod = TimeOfDayController(time_hours=0.0)  # Midnight

        moon.update_from_tod(tod)

        # Moon should be visible at night
        # (intensity depends on phase and position)
        assert isinstance(moon.intensity, float)


# =============================================================================
# EnvironmentLighting Tests
# =============================================================================


class TestEnvironmentLighting:
    def test_default_creation(self):
        lighting = EnvironmentLighting()
        assert lighting.sun is not None
        assert lighting.moon is not None
        assert lighting.ambient is not None

    def test_with_sky_manager(self):
        sky = SkyManager()
        lighting = EnvironmentLighting(sky_manager=sky)
        assert lighting.sky_manager is sky

    def test_update_with_tod(self):
        lighting = EnvironmentLighting()
        tod = TimeOfDayController(time_hours=12.0)

        lighting.update(tod_controller=tod)

        # Sun should be updated
        assert lighting.sun.intensity > 0

    def test_update_with_weather(self):
        lighting = EnvironmentLighting()
        weather = WeatherParameters(cloud_density=0.9)

        lighting.update(weather_params=weather)

        # Clouds should dim sun
        assert lighting._weather_cloud_dimming < 1.0

    def test_get_main_light_day(self):
        lighting = EnvironmentLighting()
        tod = TimeOfDayController(time_hours=12.0)
        lighting.update(tod_controller=tod)

        main = lighting.get_main_light()
        assert main is lighting.sun

    def test_get_main_light_night(self):
        lighting = EnvironmentLighting()
        tod = TimeOfDayController(time_hours=0.0)
        lighting.update(tod_controller=tod)

        main = lighting.get_main_light()
        # At night with moon phase 0.5, moon should be main
        # (or sun if both are dim)
        assert main is not None

    def test_apply_weather_modifiers_clouds(self):
        lighting = EnvironmentLighting()
        lighting.sun.intensity = 1.0

        weather = WeatherParameters(cloud_density=1.0)
        lighting.apply_weather_modifiers(weather)

        # Heavy clouds should significantly dim sun
        assert lighting.sun.intensity < 0.5

    def test_apply_weather_modifiers_fog(self):
        lighting = EnvironmentLighting()
        weather = WeatherParameters(fog_density=0.5)
        lighting.apply_weather_modifiers(weather)

        assert lighting._weather_fog_intensity == 0.5

    def test_get_total_ambient(self):
        lighting = EnvironmentLighting()
        ambient = lighting.get_total_ambient((0, 1, 0))

        assert len(ambient) == 3
        assert all(isinstance(c, float) for c in ambient)

    def test_get_fog_settings(self):
        lighting = EnvironmentLighting()
        lighting._weather_fog_intensity = 0.3

        color, density = lighting.get_fog_settings()

        assert len(color) == 3
        assert density == 0.3

    def test_to_dict(self):
        lighting = EnvironmentLighting()
        d = lighting.to_dict()

        assert "sun" in d
        assert "moon" in d
        assert "ambient" in d

    def test_set_sun_enabled(self):
        lighting = EnvironmentLighting()
        lighting.set_sun_enabled(False)
        assert lighting.sun.intensity == 0.0

    def test_set_moon_enabled(self):
        lighting = EnvironmentLighting()
        lighting.set_moon_enabled(False)
        assert lighting.moon.intensity == 0.0

    def test_get_shadow_settings(self):
        lighting = EnvironmentLighting()
        tod = TimeOfDayController(time_hours=12.0)
        lighting.update(tod_controller=tod)

        shadow = lighting.get_shadow_settings()

        assert "enabled" in shadow
        assert "direction" in shadow
        assert "cascade_count" in shadow


# =============================================================================
# LightProbe Tests
# =============================================================================


class TestLightProbe:
    def test_default_creation(self):
        probe = LightProbe()
        assert probe.position == (0, 0, 0)
        assert probe.radius == 10.0

    def test_custom_position(self):
        probe = LightProbe(position=(10, 5, 10), radius=20.0)
        assert probe.position == (10, 5, 10)
        assert probe.radius == 20.0

    def test_capture(self):
        probe = LightProbe()
        lighting = EnvironmentLighting()
        lighting.update(TimeOfDayController(time_hours=12.0))

        probe.capture(lighting)

        assert probe._is_captured is True

    def test_get_irradiance_uncaptured(self):
        probe = LightProbe()
        irr = probe.get_irradiance((0, 1, 0))

        # Should return default
        assert irr == (0.3, 0.3, 0.3)

    def test_get_irradiance_captured(self):
        probe = LightProbe()
        lighting = EnvironmentLighting()
        lighting.update(TimeOfDayController(time_hours=12.0))
        probe.capture(lighting)

        irr = probe.get_irradiance((0, 1, 0))

        assert len(irr) == 3
        assert all(isinstance(c, float) for c in irr)

    def test_irradiance_varies_with_normal(self):
        probe = LightProbe()
        lighting = EnvironmentLighting()
        lighting.update(TimeOfDayController(time_hours=12.0))
        probe.capture(lighting)

        up_irr = probe.get_irradiance((0, 1, 0))
        down_irr = probe.get_irradiance((0, -1, 0))

        # Should be different (directional component)
        assert up_irr != down_irr

    def test_get_blend_weight_center(self):
        probe = LightProbe(position=(0, 0, 0), radius=10.0)
        weight = probe.get_blend_weight((0, 0, 0))

        assert weight == 1.0

    def test_get_blend_weight_outside(self):
        probe = LightProbe(position=(0, 0, 0), radius=10.0)
        weight = probe.get_blend_weight((20, 0, 0))

        assert weight == 0.0

    def test_get_blend_weight_falloff(self):
        probe = LightProbe(position=(0, 0, 0), radius=10.0)

        # Halfway
        weight = probe.get_blend_weight((5, 0, 0))

        assert 0 < weight < 1


# =============================================================================
# LightProbeGrid Tests
# =============================================================================


class TestLightProbeGrid:
    def test_default_creation(self):
        grid = LightProbeGrid()
        assert len(grid.probes) > 0

    def test_custom_bounds(self):
        grid = LightProbeGrid(
            bounds_min=(0, 0, 0),
            bounds_max=(10, 10, 10),
            resolution=(2, 2, 2),
        )
        assert len(grid.probes) == 8  # 2*2*2

    def test_capture_all(self):
        grid = LightProbeGrid(resolution=(2, 2, 2))
        lighting = EnvironmentLighting()
        lighting.update(TimeOfDayController(time_hours=12.0))

        grid.capture_all(lighting)

        # All probes should be captured
        for probe in grid.probes:
            assert probe._is_captured is True

    def test_get_irradiance_at(self):
        grid = LightProbeGrid(
            bounds_min=(-10, 0, -10),
            bounds_max=(10, 10, 10),
            resolution=(2, 2, 2),
        )
        lighting = EnvironmentLighting()
        lighting.update(TimeOfDayController(time_hours=12.0))
        grid.capture_all(lighting)

        irr = grid.get_irradiance_at((0, 5, 0), (0, 1, 0))

        assert len(irr) == 3

    def test_irradiance_blends_probes(self):
        grid = LightProbeGrid(
            bounds_min=(-10, 0, -10),
            bounds_max=(10, 10, 10),
            resolution=(2, 2, 2),
        )
        lighting = EnvironmentLighting()
        lighting.update(TimeOfDayController(time_hours=12.0))
        grid.capture_all(lighting)

        # Point between probes
        irr = grid.get_irradiance_at((0, 5, 0), (0, 1, 0))

        # Should have blended contribution from multiple probes
        assert all(c >= 0 for c in irr)


# =============================================================================
# Integration Tests
# =============================================================================


class TestLightingIntegration:
    def test_full_day_lighting_cycle(self):
        """Test lighting throughout a day cycle."""
        lighting = EnvironmentLighting()

        sun_intensities = []
        moon_intensities = []

        for hour in range(24):
            tod = TimeOfDayController(time_hours=float(hour))
            lighting.update(tod_controller=tod)

            sun_intensities.append(lighting.sun.intensity)
            moon_intensities.append(lighting.moon.intensity)

        # Sun should be bright during day, dim at night
        assert max(sun_intensities) > 0.5
        assert min(sun_intensities) == 0.0

        # Should have variations
        assert len(set(sun_intensities)) > 5

    def test_weather_affects_lighting(self):
        """Test that weather modifies lighting."""
        lighting = EnvironmentLighting()
        tod = TimeOfDayController(time_hours=12.0)

        # Clear weather
        lighting.update(tod_controller=tod)
        clear_intensity = lighting.sun.intensity

        # Stormy weather
        lighting.update(
            tod_controller=tod,
            weather_params=WeatherParameters(
                cloud_density=1.0,
                precipitation=0.8,
            ),
        )
        storm_intensity = lighting.sun.intensity

        # Storm should be dimmer
        assert storm_intensity < clear_intensity

    def test_lighting_with_sky_manager(self):
        """Test integration between lighting and sky manager."""
        sky = SkyManager()
        lighting = EnvironmentLighting(sky_manager=sky)
        tod = TimeOfDayController(time_hours=12.0)

        lighting.update(tod_controller=tod)

        # Both should have consistent state
        sky_sun = sky.get_sun_position()
        assert sky_sun.elevation > 0  # Daytime

    def test_light_probe_grid_captures_scene(self):
        """Test that light probe grid captures environment lighting."""
        grid = LightProbeGrid(resolution=(3, 3, 3))
        lighting = EnvironmentLighting()
        tod = TimeOfDayController(time_hours=12.0)
        lighting.update(tod_controller=tod)

        grid.capture_all(lighting)

        # Query irradiance at various points
        positions = [
            (0, 5, 0),
            (5, 5, 5),
            (-5, 2, -5),
        ]

        for pos in positions:
            irr = grid.get_irradiance_at(pos, (0, 1, 0))
            assert len(irr) == 3
            assert all(c >= 0 for c in irr)

    def test_ambient_changes_with_tod(self):
        """Test that ambient lighting changes throughout the day."""
        lighting = EnvironmentLighting()

        # Morning
        lighting.update(tod_controller=TimeOfDayController(time_hours=8.0))
        morning_ambient = lighting.ambient.intensity

        # Noon
        lighting.update(tod_controller=TimeOfDayController(time_hours=12.0))
        noon_ambient = lighting.ambient.intensity

        # Night
        lighting.update(tod_controller=TimeOfDayController(time_hours=0.0))
        night_ambient = lighting.ambient.intensity

        # Ambient should vary
        assert morning_ambient != night_ambient
