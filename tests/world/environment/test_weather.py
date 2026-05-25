"""
Tests for the weather system (weather.py).

Tests weather state machine, transitions, parameters,
regional weather, and the complete weather system.
"""

import pytest
import math

from engine.world.environment.weather import (
    WeatherType,
    WeatherParameters,
    WeatherPreset,
    WeatherTransition,
    WeatherZone,
    WeatherStateMachine,
    RegionalWeather,
    WeatherSystem,
)


# =============================================================================
# WeatherParameters Tests
# =============================================================================


class TestWeatherParameters:
    def test_default_creation(self):
        params = WeatherParameters()
        assert params.precipitation == 0.0
        assert params.wind_speed == 5.0
        assert params.temperature == 20.0

    def test_custom_parameters(self):
        params = WeatherParameters(
            precipitation=0.8,
            wind_speed=15.0,
            temperature=-5.0,
        )
        assert params.precipitation == 0.8
        assert params.wind_speed == 15.0
        assert params.temperature == -5.0

    def test_lerp_basic(self):
        p1 = WeatherParameters(precipitation=0.0, temperature=20.0)
        p2 = WeatherParameters(precipitation=1.0, temperature=0.0)

        result = p1.lerp(p2, 0.5)
        assert result.precipitation == 0.5
        assert result.temperature == 10.0

    def test_lerp_at_zero(self):
        p1 = WeatherParameters(precipitation=0.2)
        p2 = WeatherParameters(precipitation=0.8)

        result = p1.lerp(p2, 0.0)
        assert result.precipitation == 0.2

    def test_lerp_at_one(self):
        p1 = WeatherParameters(precipitation=0.2)
        p2 = WeatherParameters(precipitation=0.8)

        result = p1.lerp(p2, 1.0)
        assert result.precipitation == 0.8

    def test_lerp_angle_wrapping(self):
        p1 = WeatherParameters(wind_direction=350.0)
        p2 = WeatherParameters(wind_direction=10.0)

        result = p1.lerp(p2, 0.5)
        # Should take shortest path (through 0)
        assert 355 <= result.wind_direction or result.wind_direction <= 5

    def test_lerp_clamped(self):
        p1 = WeatherParameters(precipitation=0.0)
        p2 = WeatherParameters(precipitation=1.0)

        # Out of range values should be clamped
        result_under = p1.lerp(p2, -0.5)
        result_over = p1.lerp(p2, 1.5)

        assert result_under.precipitation == 0.0
        assert result_over.precipitation == 1.0

    def test_to_dict(self):
        params = WeatherParameters(precipitation=0.5, wind_speed=10.0)
        d = params.to_dict()

        assert d["precipitation"] == 0.5
        assert d["wind_speed"] == 10.0
        assert "temperature" in d
        assert "cloud_density" in d

    def test_from_dict(self):
        d = {"precipitation": 0.7, "temperature": 15.0}
        params = WeatherParameters.from_dict(d)

        assert params.precipitation == 0.7
        assert params.temperature == 15.0


# =============================================================================
# WeatherPreset Tests
# =============================================================================


class TestWeatherPreset:
    def test_preset_creation(self):
        preset = WeatherPreset(
            weather_type=WeatherType.RAIN,
            parameters=WeatherParameters(precipitation=0.6),
            particle_system_id="particles/rain",
        )
        assert preset.weather_type == WeatherType.RAIN
        assert preset.parameters.precipitation == 0.6

    def test_default_presets_created(self):
        presets = WeatherPreset.create_default_presets()

        assert WeatherType.CLEAR in presets
        assert WeatherType.RAIN in presets
        assert WeatherType.STORM in presets
        assert WeatherType.SNOW in presets

    def test_clear_preset_has_low_precipitation(self):
        presets = WeatherPreset.create_default_presets()
        clear = presets[WeatherType.CLEAR]

        assert clear.parameters.precipitation == 0.0
        assert clear.parameters.cloud_density < 0.3

    def test_storm_preset_has_high_precipitation(self):
        presets = WeatherPreset.create_default_presets()
        storm = presets[WeatherType.STORM]

        assert storm.parameters.precipitation > 0.8
        assert storm.parameters.wind_speed > 15
        assert storm.parameters.lightning_frequency > 0

    def test_snow_preset_has_cold_temperature(self):
        presets = WeatherPreset.create_default_presets()
        snow = presets[WeatherType.SNOW]

        assert snow.parameters.temperature < 0
        assert snow.parameters.precipitation_type == "snow"


# =============================================================================
# WeatherTransition Tests
# =============================================================================


class TestWeatherTransition:
    def test_transition_creation(self):
        transition = WeatherTransition(
            from_type=WeatherType.CLEAR,
            to_type=WeatherType.RAIN,
            from_params=WeatherParameters(),
            to_params=WeatherParameters(precipitation=0.6),
            duration=60.0,
        )
        assert transition.from_type == WeatherType.CLEAR
        assert transition.to_type == WeatherType.RAIN
        assert transition.duration == 60.0

    def test_progress_calculation(self):
        transition = WeatherTransition(
            from_type=WeatherType.CLEAR,
            to_type=WeatherType.RAIN,
            from_params=WeatherParameters(),
            to_params=WeatherParameters(precipitation=0.6),
            duration=100.0,
            elapsed=50.0,
        )
        assert transition.progress == 0.5

    def test_progress_capped_at_one(self):
        transition = WeatherTransition(
            from_type=WeatherType.CLEAR,
            to_type=WeatherType.RAIN,
            from_params=WeatherParameters(),
            to_params=WeatherParameters(precipitation=0.6),
            duration=100.0,
            elapsed=150.0,
        )
        assert transition.progress == 1.0

    def test_is_complete(self):
        transition = WeatherTransition(
            from_type=WeatherType.CLEAR,
            to_type=WeatherType.RAIN,
            from_params=WeatherParameters(),
            to_params=WeatherParameters(precipitation=0.6),
            duration=100.0,
            elapsed=100.0,
        )
        assert transition.is_complete is True

    def test_blend_parameters(self):
        p_from = WeatherParameters(precipitation=0.0)
        p_to = WeatherParameters(precipitation=1.0)

        transition = WeatherTransition(
            from_type=WeatherType.CLEAR,
            to_type=WeatherType.RAIN,
            from_params=p_from,
            to_params=p_to,
            duration=100.0,
            elapsed=50.0,
            easing="linear",
        )

        blended = transition.blend_parameters()
        assert 0.4 <= blended.precipitation <= 0.6  # ~0.5 with linear

    def test_smoothstep_easing(self):
        p_from = WeatherParameters(precipitation=0.0)
        p_to = WeatherParameters(precipitation=1.0)

        transition = WeatherTransition(
            from_type=WeatherType.CLEAR,
            to_type=WeatherType.RAIN,
            from_params=p_from,
            to_params=p_to,
            duration=100.0,
            elapsed=50.0,
            easing="smoothstep",
        )

        eased = transition.get_eased_progress()
        # Smoothstep at 0.5 = 0.5 (inflection point)
        assert abs(eased - 0.5) < 0.01

    def test_update(self):
        transition = WeatherTransition(
            from_type=WeatherType.CLEAR,
            to_type=WeatherType.RAIN,
            from_params=WeatherParameters(),
            to_params=WeatherParameters(precipitation=0.6),
            duration=100.0,
        )
        transition.update(25.0)
        assert transition.elapsed == 25.0
        assert transition.progress == 0.25


# =============================================================================
# WeatherStateMachine Tests
# =============================================================================


class TestWeatherStateMachine:
    def test_initial_state(self):
        machine = WeatherStateMachine(initial_type=WeatherType.CLEAR)
        assert machine.current_type == WeatherType.CLEAR

    def test_current_parameters_from_preset(self):
        machine = WeatherStateMachine(initial_type=WeatherType.CLEAR)
        params = machine.get_current_parameters()

        # Should match clear preset
        assert params.precipitation == 0.0

    def test_can_transition_to_valid(self):
        machine = WeatherStateMachine(initial_type=WeatherType.CLEAR)
        # Clear can transition to cloudy
        assert machine.can_transition_to(WeatherType.CLOUDY) is True

    def test_can_transition_to_invalid(self):
        machine = WeatherStateMachine(initial_type=WeatherType.CLEAR)
        # Clear cannot directly transition to storm
        assert machine.can_transition_to(WeatherType.STORM) is False

    def test_can_transition_to_self(self):
        machine = WeatherStateMachine(initial_type=WeatherType.CLEAR)
        assert machine.can_transition_to(WeatherType.CLEAR) is False

    def test_start_transition(self):
        machine = WeatherStateMachine(initial_type=WeatherType.CLEAR)
        result = machine.start_transition(WeatherType.CLOUDY, duration=30.0)

        assert result is True
        assert machine.is_transitioning is True
        assert machine.target_type == WeatherType.CLOUDY

    def test_start_transition_invalid(self):
        machine = WeatherStateMachine(initial_type=WeatherType.CLEAR)
        result = machine.start_transition(WeatherType.STORM, duration=30.0)

        assert result is False
        assert machine.is_transitioning is False

    def test_start_transition_forced(self):
        machine = WeatherStateMachine(initial_type=WeatherType.CLEAR)
        result = machine.start_transition(WeatherType.STORM, duration=30.0, force=True)

        assert result is True
        assert machine.is_transitioning is True

    def test_update_progresses_transition(self):
        machine = WeatherStateMachine(initial_type=WeatherType.CLEAR)
        machine.start_transition(WeatherType.CLOUDY, duration=100.0)

        params = machine.update(50.0)
        assert machine.get_transition_progress() == 0.5

    def test_update_completes_transition(self):
        machine = WeatherStateMachine(initial_type=WeatherType.CLEAR)
        machine.start_transition(WeatherType.CLOUDY, duration=100.0)

        machine.update(100.0)  # Complete transition

        assert machine.is_transitioning is False
        assert machine.current_type == WeatherType.CLOUDY

    def test_set_weather_instant(self):
        machine = WeatherStateMachine(initial_type=WeatherType.CLEAR)
        result = machine.set_weather_instant(WeatherType.STORM)

        assert result is True
        assert machine.current_type == WeatherType.STORM
        assert machine.is_transitioning is False

    def test_weather_changed_callback(self):
        machine = WeatherStateMachine(initial_type=WeatherType.CLEAR)
        changes = []
        machine.add_weather_changed_callback(lambda old, new: changes.append((old, new)))

        machine.set_weather_instant(WeatherType.CLOUDY)

        assert len(changes) == 1
        assert changes[0] == (WeatherType.CLEAR, WeatherType.CLOUDY)

    def test_transition_start_callback(self):
        machine = WeatherStateMachine(initial_type=WeatherType.CLEAR)
        starts = []
        machine.add_transition_start_callback(lambda t: starts.append(t))

        machine.start_transition(WeatherType.CLOUDY, duration=30.0)

        assert len(starts) == 1
        assert starts[0].to_type == WeatherType.CLOUDY

    def test_transition_complete_callback(self):
        machine = WeatherStateMachine(initial_type=WeatherType.CLEAR)
        completions = []
        machine.add_transition_complete_callback(lambda t: completions.append(t))

        machine.start_transition(WeatherType.CLOUDY, duration=30.0)
        machine.update(30.0)  # Complete

        assert len(completions) == 1
        assert completions[0].to_type == WeatherType.CLOUDY

    def test_get_allowed_transitions(self):
        machine = WeatherStateMachine(initial_type=WeatherType.RAIN)
        allowed = machine.get_allowed_transitions()

        assert WeatherType.CLOUDY in allowed
        assert WeatherType.STORM in allowed

    def test_get_current_preset(self):
        machine = WeatherStateMachine(initial_type=WeatherType.STORM)
        preset = machine.get_current_preset()

        assert preset.weather_type == WeatherType.STORM
        assert preset.particle_system_id is not None


# =============================================================================
# WeatherZone Tests
# =============================================================================


class TestWeatherZone:
    def test_zone_creation(self):
        zone = WeatherZone(
            zone_id="desert",
            center=(100, 0, 100),
            radius=50.0,
            weather_type=WeatherType.CLEAR,
            parameters=WeatherParameters(temperature=40.0),
        )
        assert zone.zone_id == "desert"
        assert zone.center == (100, 0, 100)

    def test_contains_point_inside(self):
        zone = WeatherZone(
            zone_id="test",
            center=(0, 0, 0),
            radius=10.0,
            weather_type=WeatherType.CLEAR,
            parameters=WeatherParameters(),
        )
        assert zone.contains_point((0, 0, 0)) is True
        assert zone.contains_point((5, 0, 0)) is True

    def test_contains_point_outside(self):
        zone = WeatherZone(
            zone_id="test",
            center=(0, 0, 0),
            radius=10.0,
            weather_type=WeatherType.CLEAR,
            parameters=WeatherParameters(),
        )
        assert zone.contains_point((15, 0, 0)) is False

    def test_blend_weight_center(self):
        zone = WeatherZone(
            zone_id="test",
            center=(0, 0, 0),
            radius=10.0,
            weather_type=WeatherType.CLEAR,
            parameters=WeatherParameters(),
            blend_radius=2.0,
        )
        weight = zone.get_blend_weight((0, 0, 0))
        assert weight == 1.0

    def test_blend_weight_edge(self):
        zone = WeatherZone(
            zone_id="test",
            center=(0, 0, 0),
            radius=10.0,
            weather_type=WeatherType.CLEAR,
            parameters=WeatherParameters(),
            blend_radius=2.0,
        )
        weight = zone.get_blend_weight((10, 0, 0))
        assert weight == 0.0

    def test_blend_weight_in_blend_zone(self):
        zone = WeatherZone(
            zone_id="test",
            center=(0, 0, 0),
            radius=10.0,
            weather_type=WeatherType.CLEAR,
            parameters=WeatherParameters(),
            blend_radius=2.0,
        )
        # 9 units from center, in blend zone (8-10)
        weight = zone.get_blend_weight((9, 0, 0))
        assert 0 < weight < 1


# =============================================================================
# RegionalWeather Tests
# =============================================================================


class TestRegionalWeather:
    def test_creation(self):
        regional = RegionalWeather()
        assert len(regional.zones) == 0

    def test_add_zone(self):
        regional = RegionalWeather()
        zone = WeatherZone(
            zone_id="test",
            center=(0, 0, 0),
            radius=10.0,
            weather_type=WeatherType.CLEAR,
            parameters=WeatherParameters(),
        )
        regional.add_zone(zone)
        assert len(regional.zones) == 1

    def test_remove_zone(self):
        regional = RegionalWeather()
        zone = WeatherZone(
            zone_id="test",
            center=(0, 0, 0),
            radius=10.0,
            weather_type=WeatherType.CLEAR,
            parameters=WeatherParameters(),
        )
        regional.add_zone(zone)
        removed = regional.remove_zone("test")
        assert removed is zone
        assert len(regional.zones) == 0

    def test_get_weather_at_no_zones(self):
        regional = RegionalWeather(
            default_params=WeatherParameters(temperature=25.0)
        )
        params = regional.get_weather_at((0, 0, 0))
        assert params.temperature == 25.0

    def test_get_weather_at_in_zone(self):
        regional = RegionalWeather()
        zone = WeatherZone(
            zone_id="hot",
            center=(0, 0, 0),
            radius=10.0,
            weather_type=WeatherType.CLEAR,
            parameters=WeatherParameters(temperature=40.0),
        )
        regional.add_zone(zone)

        params = regional.get_weather_at((0, 0, 0))
        assert params.temperature == 40.0

    def test_get_weather_at_blends_zones(self):
        regional = RegionalWeather(
            default_params=WeatherParameters(temperature=20.0)
        )

        zone1 = WeatherZone(
            zone_id="hot",
            center=(0, 0, 0),
            radius=10.0,
            weather_type=WeatherType.CLEAR,
            parameters=WeatherParameters(temperature=40.0),
            blend_radius=5.0,
            priority=1,
        )
        regional.add_zone(zone1)

        # At blend boundary, should be interpolated
        params = regional.get_weather_at((8, 0, 0))
        assert 20 < params.temperature < 40

    def test_get_zones_at(self):
        regional = RegionalWeather()
        zone1 = WeatherZone(
            zone_id="zone1",
            center=(0, 0, 0),
            radius=10.0,
            weather_type=WeatherType.CLEAR,
            parameters=WeatherParameters(),
        )
        zone2 = WeatherZone(
            zone_id="zone2",
            center=(5, 0, 0),
            radius=10.0,
            weather_type=WeatherType.RAIN,
            parameters=WeatherParameters(),
        )
        regional.add_zone(zone1)
        regional.add_zone(zone2)

        # Point in both zones
        zones = regional.get_zones_at((3, 0, 0))
        assert len(zones) == 2


# =============================================================================
# WeatherSystem Tests
# =============================================================================


class TestWeatherSystem:
    def test_creation(self):
        system = WeatherSystem()
        assert system.current_type == WeatherType.CLEAR

    def test_creation_with_regional(self):
        system = WeatherSystem(use_regional=True)
        assert system.regional_weather is not None

    def test_update(self):
        system = WeatherSystem()
        system.transition_to(WeatherType.CLOUDY, duration=30.0)
        params = system.update(15.0)

        # Should be mid-transition
        assert system.state_machine.is_transitioning is True

    def test_transition_to(self):
        system = WeatherSystem()
        result = system.transition_to(WeatherType.CLOUDY, duration=30.0)
        assert result is True

    def test_set_weather(self):
        system = WeatherSystem()
        result = system.set_weather(WeatherType.STORM)

        assert result is True
        assert system.current_type == WeatherType.STORM

    def test_get_weather_at_without_regional(self):
        system = WeatherSystem(use_regional=False)
        params = system.get_weather_at((100, 0, 100))

        # Should return global weather
        assert params == system.state_machine.get_current_parameters()

    def test_get_weather_at_with_regional(self):
        system = WeatherSystem(use_regional=True)
        system.regional_weather.add_zone(WeatherZone(
            zone_id="hot",
            center=(0, 0, 0),
            radius=10.0,
            weather_type=WeatherType.CLEAR,
            parameters=WeatherParameters(temperature=40.0),
        ))

        params = system.get_weather_at((0, 0, 0))
        assert params.temperature == 40.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestWeatherIntegration:
    def test_full_weather_cycle(self):
        """Test a complete weather transition cycle."""
        system = WeatherSystem()

        # Start clear
        assert system.current_type == WeatherType.CLEAR

        # Transition to cloudy
        system.transition_to(WeatherType.CLOUDY, duration=10.0)
        for _ in range(10):
            system.update(1.0)
        assert system.current_type == WeatherType.CLOUDY

        # Transition to rain
        system.state_machine.start_transition(WeatherType.RAIN, duration=10.0)
        for _ in range(10):
            system.update(1.0)
        assert system.current_type == WeatherType.RAIN

    def test_interrupted_transition(self):
        """Test changing target during transition."""
        machine = WeatherStateMachine(initial_type=WeatherType.CLEAR)

        # Start transition to cloudy
        machine.start_transition(WeatherType.CLOUDY, duration=100.0)
        machine.update(30.0)  # Partial progress

        # Start new transition to fog
        machine.start_transition(WeatherType.FOG, duration=50.0, force=True)

        # Should now be transitioning to fog
        assert machine.target_type == WeatherType.FOG
        assert machine.is_transitioning is True

    def test_weather_affects_parameters_realistically(self):
        """Test that weather types have realistic parameter differences."""
        machine = WeatherStateMachine()

        machine.set_weather_instant(WeatherType.CLEAR)
        clear_params = machine.get_current_parameters()

        machine.set_weather_instant(WeatherType.STORM)
        storm_params = machine.get_current_parameters()

        # Storm should have more precipitation
        assert storm_params.precipitation > clear_params.precipitation

        # Storm should have higher wind
        assert storm_params.wind_speed > clear_params.wind_speed

        # Storm should have more clouds
        assert storm_params.cloud_density > clear_params.cloud_density
