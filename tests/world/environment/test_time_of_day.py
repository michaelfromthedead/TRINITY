"""
Tests for the time of day system (time_of_day.py).

Tests time progression, sun position calculation, lighting
interpolation, and period detection.
"""

import pytest
import math

from engine.world.environment.time_of_day import (
    TimeOfDayPeriod,
    SunPosition,
    TODLighting,
    TODKeyframe,
    TODCurve,
    TimeOfDayPreset,
    TimeOfDayController,
    PERIOD_BOUNDARIES,
)


# =============================================================================
# SunPosition Tests
# =============================================================================


class TestSunPosition:
    def test_default_creation(self):
        pos = SunPosition()
        assert pos.azimuth == 180.0
        assert pos.elevation == 45.0

    def test_direction_at_noon(self):
        pos = SunPosition(azimuth=180.0, elevation=90.0)  # Directly overhead
        direction = pos.direction
        # Should point straight up
        assert abs(direction[1] - 1.0) < 0.01
        assert abs(direction[0]) < 0.01
        assert abs(direction[2]) < 0.01

    def test_direction_at_horizon(self):
        pos = SunPosition(azimuth=180.0, elevation=0.0)  # South horizon
        direction = pos.direction
        # Should point south (azimuth 180 = south = -Z after cos(180) = -1)
        # In this coordinate system: x = sin(az)*cos(el), y = sin(el), z = cos(az)*cos(el)
        assert abs(direction[1]) < 0.01  # No vertical component (elevation=0)
        # At azimuth=180, cos(180) = -1, so z = -1
        assert direction[2] < 0  # Pointing south in this coordinate system

    def test_is_day(self):
        day_pos = SunPosition(elevation=45.0)
        night_pos = SunPosition(elevation=-45.0)

        assert day_pos.is_day is True
        assert night_pos.is_day is False

    def test_is_golden_hour(self):
        golden = SunPosition(elevation=10.0)
        not_golden = SunPosition(elevation=45.0)
        night = SunPosition(elevation=-5.0)

        assert golden.is_golden_hour is True
        assert not_golden.is_golden_hour is False
        assert night.is_golden_hour is False

    def test_lerp(self):
        pos1 = SunPosition(azimuth=90.0, elevation=30.0)
        pos2 = SunPosition(azimuth=270.0, elevation=60.0)

        result = pos1.lerp(pos2, 0.5)

        # Elevation should be midpoint
        assert abs(result.elevation - 45.0) < 0.01

    def test_lerp_azimuth_shortest_path(self):
        pos1 = SunPosition(azimuth=350.0, elevation=30.0)
        pos2 = SunPosition(azimuth=10.0, elevation=30.0)

        result = pos1.lerp(pos2, 0.5)

        # Should go through 0, not 180
        assert result.azimuth > 350 or result.azimuth < 10


# =============================================================================
# TODLighting Tests
# =============================================================================


class TestTODLighting:
    def test_default_creation(self):
        lighting = TODLighting()
        assert lighting.sun_intensity == 1.0
        assert lighting.ambient_intensity == 0.3

    def test_lerp_colors(self):
        l1 = TODLighting(sun_color=(1.0, 0.0, 0.0))
        l2 = TODLighting(sun_color=(0.0, 1.0, 0.0))

        result = l1.lerp(l2, 0.5)

        assert abs(result.sun_color[0] - 0.5) < 0.01
        assert abs(result.sun_color[1] - 0.5) < 0.01
        assert abs(result.sun_color[2] - 0.0) < 0.01

    def test_lerp_intensities(self):
        l1 = TODLighting(sun_intensity=0.0, ambient_intensity=0.0)
        l2 = TODLighting(sun_intensity=1.0, ambient_intensity=1.0)

        result = l1.lerp(l2, 0.5)

        assert abs(result.sun_intensity - 0.5) < 0.01
        assert abs(result.ambient_intensity - 0.5) < 0.01

    def test_lerp_clamped(self):
        l1 = TODLighting(sun_intensity=0.0)
        l2 = TODLighting(sun_intensity=1.0)

        result_under = l1.lerp(l2, -0.5)
        result_over = l1.lerp(l2, 1.5)

        assert result_under.sun_intensity == 0.0
        assert result_over.sun_intensity == 1.0

    def test_to_dict(self):
        lighting = TODLighting(sun_intensity=0.8)
        d = lighting.to_dict()

        assert d["sun_intensity"] == 0.8
        assert "sun_color" in d
        assert "ambient_color" in d


# =============================================================================
# TODKeyframe Tests
# =============================================================================


class TestTODKeyframe:
    def test_creation(self):
        kf = TODKeyframe(time_hours=12.0, lighting=TODLighting())
        assert kf.time_hours == 12.0

    def test_time_normalized(self):
        kf = TODKeyframe(time_hours=26.0, lighting=TODLighting())
        assert kf.time_hours == 2.0  # 26 % 24 = 2


# =============================================================================
# TODCurve Tests
# =============================================================================


class TestTODCurve:
    def test_empty_curve(self):
        curve = TODCurve()
        lighting = curve.interpolate(12.0)
        # Should return default lighting
        assert lighting is not None

    def test_single_keyframe(self):
        lighting = TODLighting(sun_intensity=0.5)
        curve = TODCurve([TODKeyframe(12.0, lighting)])

        result = curve.interpolate(12.0)
        assert result.sun_intensity == 0.5

        # Any time should return the same value
        result = curve.interpolate(6.0)
        assert result.sun_intensity == 0.5

    def test_two_keyframes_interpolation(self):
        l1 = TODLighting(sun_intensity=0.0)
        l2 = TODLighting(sun_intensity=1.0)

        curve = TODCurve([
            TODKeyframe(0.0, l1),
            TODKeyframe(12.0, l2),
        ])

        # At midpoint
        result = curve.interpolate(6.0)
        assert 0.4 < result.sun_intensity < 0.6

    def test_keyframes_sorted(self):
        l1 = TODLighting(sun_intensity=0.0)
        l2 = TODLighting(sun_intensity=1.0)

        curve = TODCurve([
            TODKeyframe(12.0, l2),
            TODKeyframe(0.0, l1),  # Added after but earlier time
        ])

        # Should be sorted by time
        assert curve.keyframes[0].time_hours == 0.0
        assert curve.keyframes[1].time_hours == 12.0

    def test_wrap_around_midnight(self):
        l_night = TODLighting(sun_intensity=0.0)
        l_day = TODLighting(sun_intensity=1.0)

        curve = TODCurve([
            TODKeyframe(6.0, l_day),
            TODKeyframe(18.0, l_night),
        ])

        # At midnight (between 18:00 and 6:00)
        result = curve.interpolate(0.0)
        # Should be interpolating between night and day
        assert 0 < result.sun_intensity < 1

    def test_add_keyframe(self):
        curve = TODCurve()
        curve.add_keyframe(TODKeyframe(12.0, TODLighting()))
        assert len(curve.keyframes) == 1

    def test_remove_keyframe(self):
        curve = TODCurve([TODKeyframe(12.0, TODLighting())])
        removed = curve.remove_keyframe(12.0)
        assert removed is not None
        assert len(curve.keyframes) == 0


# =============================================================================
# TimeOfDayPreset Tests
# =============================================================================


class TestTimeOfDayPreset:
    def test_realistic_preset(self):
        curve = TimeOfDayPreset.realistic()
        assert len(curve.keyframes) > 0

        # Should have keyframes for different times
        times = [kf.time_hours for kf in curve.keyframes]
        assert any(0 <= t <= 6 for t in times)  # Night/dawn
        assert any(10 <= t <= 14 for t in times)  # Noon
        assert any(17 <= t <= 20 for t in times)  # Sunset

    def test_stylized_preset(self):
        curve = TimeOfDayPreset.stylized()
        assert len(curve.keyframes) > 0

    def test_always_noon_preset(self):
        curve = TimeOfDayPreset.always_noon()

        # Any time should return same lighting
        l1 = curve.interpolate(0.0)
        l2 = curve.interpolate(12.0)
        l3 = curve.interpolate(20.0)

        assert l1.sun_intensity == l2.sun_intensity == l3.sun_intensity

    def test_always_night_preset(self):
        curve = TimeOfDayPreset.always_night()

        lighting = curve.interpolate(12.0)
        assert lighting.sun_intensity == 0.0
        assert lighting.moon_intensity > 0


# =============================================================================
# TimeOfDayController Tests
# =============================================================================


class TestTimeOfDayController:
    def test_default_creation(self):
        controller = TimeOfDayController()
        assert controller.time_hours == 12.0
        assert controller.time_scale == 1.0

    def test_custom_initial_time(self):
        controller = TimeOfDayController(time_hours=6.0)
        assert controller.time_hours == 6.0

    def test_time_normalized(self):
        controller = TimeOfDayController(time_hours=30.0)
        assert controller.time_hours == 6.0  # 30 % 24 = 6

    def test_update_progresses_time(self):
        controller = TimeOfDayController(time_hours=12.0, time_scale=3600.0)  # 1 hour per second
        controller.update(1.0)  # 1 second

        assert abs(controller.time_hours - 13.0) < 0.01

    def test_update_wraps_day(self):
        controller = TimeOfDayController(time_hours=23.0, time_scale=3600.0)
        controller.update(2.0)  # 2 hours

        assert controller.time_hours < 2.0  # Should have wrapped
        assert controller.day_count == 1

    def test_paused(self):
        controller = TimeOfDayController(time_hours=12.0, time_scale=3600.0)
        controller.paused = True
        controller.update(1.0)

        assert controller.time_hours == 12.0  # Didn't change

    def test_set_time(self):
        controller = TimeOfDayController()
        controller.set_time(6.0)
        assert controller.time_hours == 6.0

    def test_get_time_string(self):
        controller = TimeOfDayController(time_hours=14.5)
        time_str = controller.get_time_string()
        assert time_str == "14:30"

    def test_get_normalized_time(self):
        controller = TimeOfDayController(time_hours=12.0)
        assert controller.get_normalized_time() == 0.5

    def test_is_daytime(self):
        day_controller = TimeOfDayController(time_hours=12.0)
        night_controller = TimeOfDayController(time_hours=0.0)

        assert day_controller.is_daytime() is True
        assert night_controller.is_daytime() is False

    def test_get_period_night(self):
        controller = TimeOfDayController(time_hours=2.0)
        assert controller.get_period() == TimeOfDayPeriod.NIGHT

    def test_get_period_dawn(self):
        controller = TimeOfDayController(time_hours=5.0)
        assert controller.get_period() == TimeOfDayPeriod.DAWN

    def test_get_period_morning(self):
        controller = TimeOfDayController(time_hours=9.0)
        assert controller.get_period() == TimeOfDayPeriod.MORNING

    def test_get_period_noon(self):
        controller = TimeOfDayController(time_hours=12.0)
        assert controller.get_period() == TimeOfDayPeriod.NOON

    def test_get_period_afternoon(self):
        controller = TimeOfDayController(time_hours=15.0)
        assert controller.get_period() == TimeOfDayPeriod.AFTERNOON

    def test_get_period_sunset(self):
        controller = TimeOfDayController(time_hours=18.0)
        assert controller.get_period() == TimeOfDayPeriod.SUNSET

    def test_get_period_dusk(self):
        controller = TimeOfDayController(time_hours=20.0)
        assert controller.get_period() == TimeOfDayPeriod.DUSK

    def test_get_sun_position(self):
        controller = TimeOfDayController(time_hours=12.0, latitude=0.0)
        sun_pos = controller.get_sun_position()

        # At equator at noon, sun should be nearly overhead
        assert sun_pos.elevation > 60

    def test_sun_position_varies_with_time(self):
        controller = TimeOfDayController(latitude=45.0)

        controller.set_time(6.0)
        morning_sun = controller.get_sun_position()

        controller.set_time(12.0)
        noon_sun = controller.get_sun_position()

        controller.set_time(18.0)
        evening_sun = controller.get_sun_position()

        # Noon should be highest
        assert noon_sun.elevation > morning_sun.elevation
        assert noon_sun.elevation > evening_sun.elevation

    def test_sun_position_varies_with_latitude(self):
        equator = TimeOfDayController(time_hours=12.0, latitude=0.0)
        arctic = TimeOfDayController(time_hours=12.0, latitude=66.0)

        equator_sun = equator.get_sun_position()
        arctic_sun = arctic.get_sun_position()

        # Sun higher at equator at noon
        assert equator_sun.elevation > arctic_sun.elevation

    def test_get_lighting(self):
        controller = TimeOfDayController(time_hours=12.0)
        lighting = controller.get_lighting()

        assert lighting.sun_intensity > 0

    def test_get_sun_direction(self):
        controller = TimeOfDayController(time_hours=12.0)
        direction = controller.get_sun_direction()

        # Should be normalized
        length = math.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
        assert abs(length - 1.0) < 0.01

    def test_period_callback(self):
        controller = TimeOfDayController(time_hours=10.5, time_scale=3600.0)
        # Force the controller to record current period by calling get_period once
        _ = controller.get_period()
        # Manually set previous period
        controller._previous_period = controller.get_period()

        period_changes = []
        controller.add_period_callback(lambda old, new: period_changes.append((old, new)))

        # Advance from MORNING to NOON (boundary at 11:00)
        controller.update(1.0)  # 1 hour

        assert len(period_changes) == 1
        assert period_changes[0] == (TimeOfDayPeriod.MORNING, TimeOfDayPeriod.NOON)

    def test_day_of_year_cycles(self):
        controller = TimeOfDayController(day_of_year=365, time_hours=23.0, time_scale=3600.0)
        controller.update(2.0)  # Advance past midnight

        assert controller.day_of_year == 1  # Should wrap to 1

    def test_latitude_setter(self):
        controller = TimeOfDayController()
        controller.latitude = 100.0  # Out of range
        assert controller.latitude == 90.0  # Clamped

        controller.latitude = -100.0
        assert controller.latitude == -90.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestTimeOfDayIntegration:
    def test_full_day_cycle(self):
        """Test a complete day cycle with lighting changes."""
        controller = TimeOfDayController(time_hours=0.0, time_scale=3600.0)

        periods_seen = set()
        intensities = []

        for hour in range(25):  # 25 hours to cover wrap
            controller.update(1.0)
            periods_seen.add(controller.get_period())
            lighting = controller.get_lighting()
            intensities.append(lighting.sun_intensity)

        # Should see multiple periods
        assert len(periods_seen) >= 4

        # Intensity should vary
        assert max(intensities) > min(intensities)

    def test_lighting_matches_sun_position(self):
        """Test that lighting intensity correlates with sun elevation."""
        controller = TimeOfDayController()

        # At noon
        controller.set_time(12.0)
        noon_sun = controller.get_sun_position()
        noon_light = controller.get_lighting()

        # At night
        controller.set_time(0.0)
        night_sun = controller.get_sun_position()
        night_light = controller.get_lighting()

        # Higher sun = higher intensity
        assert noon_sun.elevation > night_sun.elevation
        assert noon_light.sun_intensity > night_light.sun_intensity

    def test_smooth_transitions(self):
        """Test that lighting transitions smoothly."""
        controller = TimeOfDayController(time_hours=5.0)  # Dawn

        prev_intensity = controller.get_lighting().sun_intensity

        for _ in range(60):  # Simulate one hour in small steps
            controller.update(60.0)  # 1 minute real time
            current_intensity = controller.get_lighting().sun_intensity

            # Changes should be gradual
            assert abs(current_intensity - prev_intensity) < 0.1

            prev_intensity = current_intensity
