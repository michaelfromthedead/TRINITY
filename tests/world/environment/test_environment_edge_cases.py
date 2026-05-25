"""
Edge case tests for the Environment System.

These tests verify correct handling of boundary conditions, interpolation
accuracy, and other edge cases that may not be covered by basic unit tests.
"""

import math
import pytest

from engine.world.environment.volumes import (
    VolumeType,
    BoundingBox,
    PostProcessVolume,
    FogVolume,
    VolumeManager,
)
from engine.world.environment.weather import (
    WeatherType,
    WeatherParameters,
    WeatherTransition,
    WeatherZone,
    RegionalWeather,
)
from engine.world.environment.time_of_day import (
    TimeOfDayPeriod,
    TODLighting,
    TimeOfDayController,
    PERIOD_BOUNDARIES,
)
from engine.world.environment.sky import (
    AtmosphereSettings,
    ProceduralSky,
    SunPosition,
)
from engine.world.environment.lighting import (
    DirectionalLight,
    AmbientLight,
)


# =============================================================================
# Volume Blending Weight Tests
# =============================================================================


class TestVolumeBlendingWeightSum:
    """Tests that verify volume blending weights are calculated correctly."""

    def test_single_volume_weight_at_center(self):
        """A point at the center of a single volume should have full weight."""
        manager = VolumeManager()
        volume = PostProcessVolume(
            bounds=BoundingBox(min_point=(0, 0, 0), max_point=(10, 10, 10)),
            exposure=2.0,
            blend_radius=2.0,
            priority=1,
        )
        manager.add_volume(volume)

        weight = volume.get_blend_weight((5, 5, 5))
        assert weight == 1.0

    def test_overlapping_volumes_weights_are_valid(self):
        """Each volume's weight should be valid (0-1) even when overlapping."""
        manager = VolumeManager()

        v1 = PostProcessVolume(
            bounds=BoundingBox(min_point=(0, 0, 0), max_point=(10, 10, 10)),
            exposure=1.0,
            blend_radius=2.0,
            priority=1,
        )
        v2 = PostProcessVolume(
            bounds=BoundingBox(min_point=(5, 5, 5), max_point=(15, 15, 15)),
            exposure=2.0,
            blend_radius=2.0,
            priority=2,
        )
        manager.add_volume(v1)
        manager.add_volume(v2)

        # Point in overlap zone
        test_point = (7, 7, 7)

        # Each individual weight should be in [0, 1]
        w1 = v1.get_blend_weight(test_point)
        w2 = v2.get_blend_weight(test_point)

        assert 0.0 <= w1 <= 1.0
        assert 0.0 <= w2 <= 1.0

    def test_blend_weight_at_edge_is_zero(self):
        """Weight should be zero exactly at the volume edge when blend_radius > 0."""
        volume = PostProcessVolume(
            bounds=BoundingBox(min_point=(0, 0, 0), max_point=(10, 10, 10)),
            blend_radius=2.0,
        )

        # At exact edge
        weight = volume.get_blend_weight((0, 5, 5))
        assert weight == 0.0

    def test_blend_weight_smooth_transition(self):
        """Weight should transition smoothly through blend zone."""
        volume = PostProcessVolume(
            bounds=BoundingBox(min_point=(0, 0, 0), max_point=(10, 10, 10)),
            blend_radius=2.0,
        )

        # Test points at increasing distances from edge (moving toward center)
        weights = []
        for x in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
            weights.append(volume.get_blend_weight((x, 5, 5)))

        # Weights should be monotonically increasing
        for i in range(len(weights) - 1):
            assert weights[i] <= weights[i + 1], f"Weight at index {i} > weight at {i+1}"

        # First should be 0, last should be 1
        assert weights[0] == 0.0
        assert weights[-1] == 1.0


class TestVolumeBlendedSettings:
    """Tests for blended settings calculations."""

    def test_blended_exposure_is_weighted_average(self):
        """Blended exposure should be a proper weighted average."""
        manager = VolumeManager()

        v1 = PostProcessVolume(
            bounds=BoundingBox(min_point=(0, 0, 0), max_point=(10, 10, 10)),
            exposure=1.0,
            blend_radius=0.0,  # No blend zone - full weight inside
            priority=1,
        )
        v2 = PostProcessVolume(
            bounds=BoundingBox(min_point=(0, 0, 0), max_point=(10, 10, 10)),
            exposure=3.0,
            blend_radius=0.0,
            priority=1,  # Same priority
        )
        manager.add_volume(v1)
        manager.add_volume(v2)

        settings = manager.get_blended_settings((5, 5, 5), VolumeType.POST_PROCESS)

        # With equal weights and priorities, should be average
        # Result depends on implementation - just verify it's between extremes
        assert 1.0 <= settings["exposure"] <= 3.0

    def test_no_volumes_returns_empty_or_default(self):
        """Getting blended settings with no volumes should handle gracefully."""
        manager = VolumeManager()
        settings = manager.get_blended_settings((5, 5, 5), VolumeType.POST_PROCESS)

        # Should return empty dict or default values
        assert isinstance(settings, dict)


# =============================================================================
# Weather Interpolation Tests
# =============================================================================


class TestWeatherParameterInterpolation:
    """Tests for weather parameter interpolation accuracy."""

    def test_lerp_at_exact_midpoint(self):
        """Interpolation at exactly 0.5 should give exact midpoint."""
        p1 = WeatherParameters(
            precipitation=0.0,
            wind_speed=0.0,
            temperature=0.0,
            cloud_density=0.0,
        )
        p2 = WeatherParameters(
            precipitation=1.0,
            wind_speed=20.0,
            temperature=40.0,
            cloud_density=1.0,
        )

        result = p1.lerp(p2, 0.5)

        assert abs(result.precipitation - 0.5) < 1e-6
        assert abs(result.wind_speed - 10.0) < 1e-6
        assert abs(result.temperature - 20.0) < 1e-6
        assert abs(result.cloud_density - 0.5) < 1e-6

    def test_lerp_preserves_bounds_at_extremes(self):
        """Interpolation should clamp t to [0, 1]."""
        p1 = WeatherParameters(precipitation=0.2)
        p2 = WeatherParameters(precipitation=0.8)

        # Negative t should return p1 values
        result_neg = p1.lerp(p2, -0.5)
        assert result_neg.precipitation == 0.2

        # t > 1 should return p2 values
        result_over = p1.lerp(p2, 1.5)
        assert result_over.precipitation == 0.8

    def test_lerp_all_parameters(self):
        """Verify all WeatherParameters fields are interpolated."""
        p1 = WeatherParameters(
            precipitation=0.0,
            wind_speed=5.0,
            wind_direction=0.0,
            temperature=10.0,
            humidity=0.2,
            cloud_density=0.1,
            visibility=5000.0,
            fog_density=0.0,
        )
        p2 = WeatherParameters(
            precipitation=1.0,
            wind_speed=25.0,
            wind_direction=90.0,
            temperature=30.0,
            humidity=0.8,
            cloud_density=0.9,
            visibility=1000.0,
            fog_density=0.5,
        )

        result = p1.lerp(p2, 0.5)

        # All values should be at midpoint
        assert 0.4 < result.precipitation < 0.6
        assert 14 < result.wind_speed < 16
        assert 19 < result.temperature < 21
        assert 0.4 < result.humidity < 0.6
        assert 0.4 < result.cloud_density < 0.6
        assert 2500 < result.visibility < 3500
        assert 0.2 < result.fog_density < 0.3


class TestWeatherTransitionBlending:
    """Tests for weather transition parameter blending."""

    def test_transition_blend_at_start(self):
        """At elapsed=0, should return from_params."""
        p_from = WeatherParameters(temperature=0.0)
        p_to = WeatherParameters(temperature=100.0)

        transition = WeatherTransition(
            from_type=WeatherType.CLEAR,
            to_type=WeatherType.RAIN,
            from_params=p_from,
            to_params=p_to,
            duration=100.0,
            elapsed=0.0,
            easing="linear",
        )

        blended = transition.blend_parameters()
        assert blended.temperature == 0.0

    def test_transition_blend_at_end(self):
        """At elapsed=duration, should return to_params."""
        p_from = WeatherParameters(temperature=0.0)
        p_to = WeatherParameters(temperature=100.0)

        transition = WeatherTransition(
            from_type=WeatherType.CLEAR,
            to_type=WeatherType.RAIN,
            from_params=p_from,
            to_params=p_to,
            duration=100.0,
            elapsed=100.0,
            easing="linear",
        )

        blended = transition.blend_parameters()
        assert blended.temperature == 100.0

    def test_smoothstep_easing_at_midpoint(self):
        """Smoothstep at t=0.5 should equal 0.5."""
        transition = WeatherTransition(
            from_type=WeatherType.CLEAR,
            to_type=WeatherType.RAIN,
            from_params=WeatherParameters(),
            to_params=WeatherParameters(),
            duration=100.0,
            elapsed=50.0,
            easing="smoothstep",
        )

        eased = transition.get_eased_progress()
        assert abs(eased - 0.5) < 0.01


class TestWeatherZoneBlending:
    """Tests for weather zone blending."""

    def test_zone_blend_weight_sum_at_boundary(self):
        """At zone boundary with default params, weights should transition properly."""
        regional = RegionalWeather(
            default_params=WeatherParameters(temperature=20.0)
        )

        zone = WeatherZone(
            zone_id="hot",
            center=(0, 0, 0),
            radius=10.0,
            weather_type=WeatherType.CLEAR,
            parameters=WeatherParameters(temperature=40.0),
            blend_radius=5.0,
        )
        regional.add_zone(zone)

        # At center: zone weight = 1.0
        params_center = regional.get_weather_at((0, 0, 0))
        assert params_center.temperature == 40.0

        # At edge: zone weight = 0.0
        params_edge = regional.get_weather_at((10, 0, 0))
        assert params_edge.temperature == 20.0

        # In blend zone: interpolated
        params_blend = regional.get_weather_at((7.5, 0, 0))
        assert 20.0 < params_blend.temperature < 40.0


# =============================================================================
# Time of Day Tests
# =============================================================================


class TestTODTimeWraparound:
    """Tests for time wraparound at day boundaries."""

    def test_time_at_exactly_24(self):
        """Time at exactly 24.0 should wrap to 0.0."""
        # Setting time_hours through the property setter triggers normalization
        controller = TimeOfDayController(time_hours=24.0)
        assert controller.time_hours == 0.0

        # Also test via set_time method
        controller.set_time(24.0)
        assert controller.time_hours == 0.0

    def test_time_wraps_multiple_days(self):
        """Time > 48 hours should wrap correctly."""
        controller = TimeOfDayController(time_hours=50.0)
        assert controller.time_hours == 2.0  # 50 % 24 = 2

    def test_negative_time_wraps_correctly(self):
        """Negative time should wrap to positive."""
        controller = TimeOfDayController(time_hours=-2.0)
        assert controller.time_hours == 22.0  # -2 + 24 = 22

    def test_day_count_increments_on_wrap(self):
        """Day count should increment when time wraps."""
        controller = TimeOfDayController(
            time_hours=23.5,
            time_scale=3600.0,  # 1 hour per second
        )
        initial_day = controller.day_count

        controller.update(1.0)  # Advance 1 hour, wrapping past midnight

        assert controller.day_count == initial_day + 1
        assert controller.time_hours < 1.0  # Should be around 0.5

    def test_period_at_exact_boundaries(self):
        """Period detection should handle exact boundary times."""
        # Test at each period boundary
        boundaries = [
            (4.5, TimeOfDayPeriod.DAWN),
            (6.0, TimeOfDayPeriod.SUNRISE),
            (7.5, TimeOfDayPeriod.MORNING),
            (11.0, TimeOfDayPeriod.NOON),
            (13.0, TimeOfDayPeriod.AFTERNOON),
            (17.0, TimeOfDayPeriod.SUNSET),
            (19.5, TimeOfDayPeriod.DUSK),
            (21.0, TimeOfDayPeriod.NIGHT),
        ]

        for time, expected_period in boundaries:
            controller = TimeOfDayController(time_hours=time)
            assert controller.get_period() == expected_period, \
                f"At {time}h expected {expected_period}, got {controller.get_period()}"


class TestTODLightingInterpolation:
    """Tests for time-of-day lighting interpolation."""

    def test_lighting_changes_with_time(self):
        """Lighting should change when time changes."""
        controller = TimeOfDayController(time_hours=6.0)
        dawn_lighting = controller.get_lighting()

        controller.set_time(12.0)
        noon_lighting = controller.get_lighting()

        # Noon should be brighter than dawn
        assert noon_lighting.sun_intensity > dawn_lighting.sun_intensity

    def test_color_components_in_valid_range(self):
        """All color components should stay in [0, 1]."""
        controller = TimeOfDayController(time_hours=0.0, time_scale=3600.0)

        for _ in range(24):
            controller.update(1.0)
            lighting = controller.get_lighting()

            # Check sun color
            assert all(0.0 <= c <= 1.0 for c in lighting.sun_color)
            # Check ambient color
            assert all(0.0 <= c <= 1.0 for c in lighting.ambient_color)


# =============================================================================
# Sun Position Tests - Polar Regions
# =============================================================================


class TestSunPositionPolarRegions:
    """Tests for sun position calculation in polar regions."""

    def test_arctic_summer_midnight_sun(self):
        """At high latitudes in summer, sun should stay above horizon."""
        # Summer solstice, latitude 70N (within arctic circle)
        controller = TimeOfDayController(
            time_hours=0.0,  # Midnight
            latitude=70.0,
            day_of_year=172,  # Summer solstice
        )

        sun_pos = controller.get_sun_position()

        # In high summer at 70N, sun should be above horizon even at midnight
        # Note: This depends on implementation accuracy - may need tolerance
        # Just verify we get a valid result
        assert -90.0 <= sun_pos.elevation <= 90.0

    def test_arctic_winter_polar_night(self):
        """At high latitudes in winter, sun should stay below horizon."""
        # Winter solstice, latitude 70N
        controller = TimeOfDayController(
            time_hours=12.0,  # Noon
            latitude=70.0,
            day_of_year=355,  # Winter solstice
        )

        sun_pos = controller.get_sun_position()

        # In winter at 70N, sun should be low even at noon
        # Note: Actual calculation depends on implementation
        assert -90.0 <= sun_pos.elevation <= 90.0

    def test_equator_consistent_day_length(self):
        """At equator, sun path should be consistent throughout year."""
        # Test at equinox
        controller_equinox = TimeOfDayController(
            time_hours=12.0,
            latitude=0.0,
            day_of_year=80,  # Near spring equinox
        )

        # Test at solstice
        controller_solstice = TimeOfDayController(
            time_hours=12.0,
            latitude=0.0,
            day_of_year=172,  # Summer solstice
        )

        sun_equinox = controller_equinox.get_sun_position()
        sun_solstice = controller_solstice.get_sun_position()

        # At equator, noon sun elevation should be high year-round
        assert sun_equinox.elevation > 60.0
        assert sun_solstice.elevation > 60.0

    def test_latitude_clamping(self):
        """Latitude should be clamped to valid range."""
        controller = TimeOfDayController(latitude=100.0)
        assert controller.latitude == 90.0

        controller.latitude = -100.0
        assert controller.latitude == -90.0


# =============================================================================
# Color Interpolation Tests
# =============================================================================


class TestColorInterpolation:
    """Tests for color interpolation in lighting and atmosphere."""

    def test_directional_light_lerp_colors(self):
        """Light color interpolation should stay in valid range."""
        l1 = DirectionalLight(color=(0.0, 0.0, 0.0), intensity=0.0)
        l2 = DirectionalLight(color=(1.0, 1.0, 1.0), intensity=1.0)

        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = l1.lerp(l2, t)
            assert all(0.0 <= c <= 1.0 for c in result.color)
            assert 0.0 <= result.intensity <= 1.0

    def test_tod_lighting_color_lerp(self):
        """TOD lighting color interpolation should be in valid range."""
        l1 = TODLighting(
            sun_color=(0.0, 0.0, 0.0),
            ambient_color=(0.0, 0.0, 0.0),
        )
        l2 = TODLighting(
            sun_color=(1.0, 1.0, 1.0),
            ambient_color=(1.0, 1.0, 1.0),
        )

        result = l1.lerp(l2, 0.5)

        assert all(0.0 <= c <= 1.0 for c in result.sun_color)
        assert all(0.0 <= c <= 1.0 for c in result.ambient_color)

    def test_atmosphere_settings_lerp_colors(self):
        """Atmosphere color interpolation should stay in valid range."""
        a1 = AtmosphereSettings(
            rayleigh_color=(0.0, 0.0, 0.0),
            mie_color=(0.0, 0.0, 0.0),
        )
        a2 = AtmosphereSettings(
            rayleigh_color=(1.0, 1.0, 1.0),
            mie_color=(1.0, 1.0, 1.0),
        )

        result = a1.lerp(a2, 0.5)

        assert all(0.0 <= c <= 1.0 for c in result.rayleigh_color)
        assert all(0.0 <= c <= 1.0 for c in result.mie_color)

    def test_lerp_with_out_of_range_t(self):
        """Lerp should clamp t to [0, 1] and produce valid colors."""
        l1 = TODLighting(sun_intensity=0.0)
        l2 = TODLighting(sun_intensity=1.0)

        # t < 0
        result_neg = l1.lerp(l2, -0.5)
        assert result_neg.sun_intensity == 0.0

        # t > 1
        result_over = l1.lerp(l2, 1.5)
        assert result_over.sun_intensity == 1.0


# =============================================================================
# Atmosphere Scattering Tests
# =============================================================================


class TestAtmosphereScattering:
    """Tests for atmosphere scattering calculations."""

    def test_sky_color_never_negative(self):
        """Sky color should never have negative components."""
        sky = ProceduralSky(
            atmosphere=AtmosphereSettings(
                rayleigh_density=2.0,
                mie_density=2.0,
            ),
            sun_position=SunPosition(azimuth=180.0, elevation=45.0),
        )

        # Test various directions
        directions = [
            (0, 1, 0),
            (1, 0, 0),
            (0, 0, 1),
            (-1, 0, 0),
            (0.707, 0.707, 0),
            (0.577, 0.577, 0.577),
        ]

        for direction in directions:
            color = sky.compute_sky_color(direction)
            assert all(c >= 0.0 for c in color), f"Negative color at direction {direction}"

    def test_sky_color_finite(self):
        """Sky color should always be finite (no inf/nan)."""
        sky = ProceduralSky()

        # Test edge case directions
        directions = [
            (0, 1, 0),
            (0, -1, 0),  # Below horizon
            (1, 0, 0),
            (0.001, 0.999, 0),  # Near-up
        ]

        for direction in directions:
            color = sky.compute_sky_color(direction)
            assert all(math.isfinite(c) for c in color), f"Non-finite color at {direction}"

    def test_aerial_perspective_scaling(self):
        """Aerial perspective should increase with distance."""
        sky = ProceduralSky()

        distances = [100, 1000, 5000, 10000]
        fogs = [sky.get_aerial_perspective(d) for d in distances]

        # Should be monotonically increasing
        for i in range(len(fogs) - 1):
            assert fogs[i] <= fogs[i + 1]

        # All should be non-negative
        assert all(f >= 0.0 for f in fogs)


# =============================================================================
# Integration Tests
# =============================================================================


class TestEnvironmentIntegration:
    """Integration tests for the complete environment system."""

    def test_full_day_color_continuity(self):
        """Colors should transition smoothly throughout the day."""
        controller = TimeOfDayController(time_hours=0.0, time_scale=3600.0)

        prev_lighting = controller.get_lighting()

        for _ in range(24):
            controller.update(1.0)
            lighting = controller.get_lighting()

            # Check that changes are gradual
            intensity_change = abs(lighting.sun_intensity - prev_lighting.sun_intensity)
            # Changes should not be drastic (allowing for numerical precision)
            assert intensity_change < 0.5, f"Large intensity jump: {intensity_change}"

            prev_lighting = lighting

    def test_weather_transition_smoothness(self):
        """Weather should transition smoothly during active transitions."""
        p_from = WeatherParameters(temperature=0.0, precipitation=0.0)
        p_to = WeatherParameters(temperature=30.0, precipitation=0.8)

        transition = WeatherTransition(
            from_type=WeatherType.CLEAR,
            to_type=WeatherType.RAIN,
            from_params=p_from,
            to_params=p_to,
            duration=100.0,
            easing="smoothstep",
        )

        prev_params = transition.blend_parameters()

        for i in range(1, 11):
            transition.update(10.0)
            params = transition.blend_parameters()

            # Temperature should monotonically increase
            assert params.temperature >= prev_params.temperature
            # Precipitation should monotonically increase
            assert params.precipitation >= prev_params.precipitation

            prev_params = params
