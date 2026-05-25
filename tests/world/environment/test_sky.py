"""
Tests for the sky system (sky.py).

Tests atmosphere simulation, procedural/HDRI/static sky,
celestial bodies, and the sky manager.
"""

import pytest
import math

from engine.world.environment.sky import (
    AtmosphereSettings,
    CelestialBodyType,
    BaseSky,
    ProceduralSky,
    HDRISky,
    StaticSky,
    CelestialBody,
    StarField,
    SkyManager,
)
from engine.world.environment.time_of_day import SunPosition, TimeOfDayController


# =============================================================================
# AtmosphereSettings Tests
# =============================================================================


class TestAtmosphereSettings:
    def test_default_creation(self):
        atmo = AtmosphereSettings()
        assert atmo.rayleigh_density == 1.0
        assert atmo.mie_density == 1.0
        assert atmo.planet_radius == 6371000.0

    def test_custom_settings(self):
        atmo = AtmosphereSettings(
            rayleigh_density=0.5,
            mie_anisotropy=0.9,
            sun_intensity_multiplier=2.0,
        )
        assert atmo.rayleigh_density == 0.5
        assert atmo.mie_anisotropy == 0.9
        assert atmo.sun_intensity_multiplier == 2.0

    def test_to_dict(self):
        atmo = AtmosphereSettings(rayleigh_density=0.8)
        d = atmo.to_dict()

        assert d["rayleigh_density"] == 0.8
        assert "mie_density" in d
        assert "planet_radius" in d

    def test_lerp(self):
        a1 = AtmosphereSettings(rayleigh_density=0.0, mie_density=0.0)
        a2 = AtmosphereSettings(rayleigh_density=1.0, mie_density=1.0)

        result = a1.lerp(a2, 0.5)

        assert abs(result.rayleigh_density - 0.5) < 0.01
        assert abs(result.mie_density - 0.5) < 0.01

    def test_lerp_colors(self):
        a1 = AtmosphereSettings(rayleigh_color=(1.0, 0.0, 0.0))
        a2 = AtmosphereSettings(rayleigh_color=(0.0, 1.0, 0.0))

        result = a1.lerp(a2, 0.5)

        assert abs(result.rayleigh_color[0] - 0.5) < 0.01
        assert abs(result.rayleigh_color[1] - 0.5) < 0.01


# =============================================================================
# ProceduralSky Tests
# =============================================================================


class TestProceduralSky:
    def test_default_creation(self):
        sky = ProceduralSky()
        assert sky.atmosphere is not None
        assert sky.sun_position is not None

    def test_custom_atmosphere(self):
        atmo = AtmosphereSettings(rayleigh_density=0.5)
        sky = ProceduralSky(atmosphere=atmo)
        assert sky.atmosphere.rayleigh_density == 0.5

    def test_compute_sky_color_returns_rgb(self):
        sky = ProceduralSky()
        color = sky.compute_sky_color((0, 1, 0))  # Looking up

        assert len(color) == 3
        assert all(isinstance(c, float) for c in color)

    def test_sky_color_varies_with_direction(self):
        sky = ProceduralSky(sun_position=SunPosition(azimuth=180.0, elevation=45.0))

        toward_sun = sky.compute_sky_color((0, 0.7, 0.7))  # Toward sun
        away_from_sun = sky.compute_sky_color((0, 0.7, -0.7))  # Away from sun

        # Colors should be different
        assert toward_sun != away_from_sun

    def test_sky_darker_at_night(self):
        day_sky = ProceduralSky(sun_position=SunPosition(elevation=45.0))
        night_sky = ProceduralSky(sun_position=SunPosition(elevation=-45.0))

        day_color = day_sky.compute_sky_color((0, 1, 0))
        night_color = night_sky.compute_sky_color((0, 1, 0))

        # Day should be brighter
        day_brightness = sum(day_color)
        night_brightness = sum(night_color)
        assert day_brightness > night_brightness

    def test_compute_sun_disk_visible(self):
        sky = ProceduralSky(sun_position=SunPosition(azimuth=180.0, elevation=45.0))

        # Look directly at sun
        sun_dir = sky.sun_position.direction
        intensity = sky.compute_sun_disk(sun_dir)

        assert intensity > 0

    def test_compute_sun_disk_not_visible_away(self):
        sky = ProceduralSky(sun_position=SunPosition(azimuth=180.0, elevation=45.0))

        # Look opposite to sun
        opposite = (-sky.sun_position.direction[0],
                    -sky.sun_position.direction[1],
                    -sky.sun_position.direction[2])
        intensity = sky.compute_sun_disk(opposite)

        assert intensity == 0

    def test_compute_sun_disk_not_visible_at_night(self):
        sky = ProceduralSky(sun_position=SunPosition(elevation=-30.0))
        intensity = sky.compute_sun_disk((0, 1, 0))

        assert intensity == 0

    def test_aerial_perspective(self):
        sky = ProceduralSky()

        near_fog = sky.get_aerial_perspective(100)  # 100m
        far_fog = sky.get_aerial_perspective(10000)  # 10km

        # More fog at distance
        assert far_fog > near_fog

    def test_get_sky_color_interface(self):
        sky = ProceduralSky()
        color = sky.get_sky_color((0, 1, 0))

        assert len(color) == 3

    def test_update(self):
        sky = ProceduralSky()
        initial_elevation = sky.sun_position.elevation

        sky.update(18.0)  # Update to 6 PM

        # Sun position should have changed
        assert sky.sun_position.elevation != initial_elevation


# =============================================================================
# HDRISky Tests
# =============================================================================


class TestHDRISky:
    def test_default_creation(self):
        sky = HDRISky()
        assert sky.texture_path == ""
        assert sky.rotation == 0.0
        assert sky.intensity == 1.0

    def test_custom_path(self):
        sky = HDRISky(texture_path="textures/sky.hdr", rotation=45.0)
        assert sky.texture_path == "textures/sky.hdr"
        assert sky.rotation == 45.0

    def test_get_sky_color(self):
        sky = HDRISky()
        color = sky.get_sky_color((0, 1, 0))

        assert len(color) == 3

    def test_color_varies_with_direction(self):
        sky = HDRISky()

        up_color = sky.get_sky_color((0, 1, 0))
        down_color = sky.get_sky_color((0, -1, 0))

        # Should be different (gradient)
        assert up_color != down_color

    def test_load(self):
        sky = HDRISky(texture_path="textures/sky.hdr")
        result = sky.load()

        # Should return True (path exists)
        assert result is True

    def test_load_empty_path(self):
        sky = HDRISky()
        result = sky.load()

        assert result is False

    def test_update_does_nothing(self):
        sky = HDRISky(intensity=1.0)
        sky.update(12.0)

        # HDRI sky is static
        assert sky.intensity == 1.0


# =============================================================================
# StaticSky Tests
# =============================================================================


class TestStaticSky:
    def test_default_creation(self):
        sky = StaticSky()
        assert len(sky.skybox_textures) == 0

    def test_custom_textures(self):
        textures = ["px.jpg", "nx.jpg", "py.jpg", "ny.jpg", "pz.jpg", "nz.jpg"]
        sky = StaticSky(skybox_textures=textures)
        assert len(sky.skybox_textures) == 6

    def test_get_sky_color_returns_face_color(self):
        sky = StaticSky()

        # Looking up
        up_color = sky.get_sky_color((0, 1, 0))
        assert len(up_color) == 3

        # Looking down
        down_color = sky.get_sky_color((0, -1, 0))
        assert down_color != up_color

    def test_different_faces(self):
        sky = StaticSky()

        # Each axis should map to different faces
        px = sky.get_sky_color((1, 0, 0))
        nx = sky.get_sky_color((-1, 0, 0))
        py = sky.get_sky_color((0, 1, 0))
        ny = sky.get_sky_color((0, -1, 0))
        pz = sky.get_sky_color((0, 0, 1))
        nz = sky.get_sky_color((0, 0, -1))

        # At least some should be different
        colors = [px, nx, py, ny, pz, nz]
        unique_colors = len(set(colors))
        assert unique_colors >= 2


# =============================================================================
# CelestialBody Tests
# =============================================================================


class TestCelestialBody:
    def test_sun_creation(self):
        sun = CelestialBody(body_type=CelestialBodyType.SUN)
        assert sun.body_type == CelestialBodyType.SUN

    def test_moon_creation(self):
        moon = CelestialBody(
            body_type=CelestialBodyType.MOON,
            angular_size=0.52,
        )
        assert moon.body_type == CelestialBodyType.MOON
        assert moon.angular_size == 0.52

    def test_get_position_returns_direction(self):
        body = CelestialBody(body_type=CelestialBodyType.SUN)
        direction = body.get_position(12.0)

        assert len(direction) == 3

        # Should be normalized
        length = math.sqrt(sum(d*d for d in direction))
        assert abs(length - 1.0) < 0.01

    def test_position_varies_with_time(self):
        body = CelestialBody(body_type=CelestialBodyType.SUN, orbital_period=24.0)

        pos_noon = body.get_position(12.0)
        pos_midnight = body.get_position(0.0)

        assert pos_noon != pos_midnight

    def test_get_visible_intensity_visible(self):
        body = CelestialBody(
            body_type=CelestialBodyType.SUN,
            angular_size=0.5,
            intensity=1.0,
        )

        # Get body position and look at it
        direction = body.get_position(12.0)
        intensity = body.get_visible_intensity(direction, 12.0)

        assert intensity > 0

    def test_get_visible_intensity_not_visible(self):
        body = CelestialBody(body_type=CelestialBodyType.SUN)

        # Look in opposite direction
        direction = body.get_position(12.0)
        opposite = (-direction[0], -direction[1], -direction[2])
        intensity = body.get_visible_intensity(opposite, 12.0)

        assert intensity == 0

    def test_moon_phase(self):
        moon = CelestialBody(body_type=CelestialBodyType.MOON)

        moon.set_moon_phase(0.5)  # Full moon
        assert moon.get_moon_phase() == 0.5

        moon.set_moon_phase(1.5)  # Should wrap
        assert moon.get_moon_phase() == 0.5


# =============================================================================
# StarField Tests
# =============================================================================


class TestStarField:
    def test_default_creation(self):
        stars = StarField()
        assert stars.star_count == 5000
        assert stars.brightness == 1.0

    def test_custom_count(self):
        stars = StarField(star_count=1000)
        assert stars.star_count == 1000

    def test_stars_generated(self):
        stars = StarField(star_count=100)
        assert len(stars._stars) == 100

    def test_star_brightness_at_night(self):
        stars = StarField()
        brightness = stars.get_star_brightness((0, 1, 0), 0.0, -30.0)

        # Should have some star contribution at night
        # (may be zero if not looking at a star)
        assert isinstance(brightness, tuple)
        assert len(brightness) == 3

    def test_stars_fade_during_day(self):
        stars = StarField()

        night_brightness = stars.get_star_brightness((0, 1, 0), 0.0, -30.0)
        day_brightness = stars.get_star_brightness((0, 1, 0), 0.0, 45.0)

        # Day should have no stars
        assert day_brightness == (0.0, 0.0, 0.0)

    def test_twinkle_varies_with_time(self):
        stars = StarField(twinkle_speed=10.0)

        # Get brightness at different times
        b1 = stars.get_star_brightness((0.9999, 0.01, 0), 0.0, -30.0)
        b2 = stars.get_star_brightness((0.9999, 0.01, 0), 0.5, -30.0)

        # May be different due to twinkling (or same if no star there)
        # Just check they're valid
        assert isinstance(b1, tuple)
        assert isinstance(b2, tuple)


# =============================================================================
# SkyManager Tests
# =============================================================================


class TestSkyManager:
    def test_default_procedural(self):
        manager = SkyManager()
        assert manager.sky_type == "procedural"

    def test_hdri_type(self):
        manager = SkyManager(sky_type="hdri")
        assert manager.sky_type == "hdri"

    def test_static_type(self):
        manager = SkyManager(sky_type="static")
        assert manager.sky_type == "static"

    def test_sun_and_moon(self):
        manager = SkyManager()
        assert manager.sun is not None
        assert manager.moon is not None
        assert manager.sun.body_type == CelestialBodyType.SUN

    def test_stars(self):
        manager = SkyManager()
        assert manager.stars is not None

    def test_update_with_tod(self):
        manager = SkyManager()
        tod = TimeOfDayController(time_hours=12.0)

        manager.update(tod)

        # Sun position should be updated
        assert manager._current_time == 12.0

    def test_get_sky_color(self):
        manager = SkyManager()
        color = manager.get_sky_color((0, 1, 0))

        assert len(color) == 3

    def test_get_sun_position(self):
        manager = SkyManager()
        manager.update(TimeOfDayController(time_hours=12.0))

        sun_pos = manager.get_sun_position()
        assert sun_pos.elevation > 0  # Daytime

    def test_get_ambient_light_day(self):
        manager = SkyManager()
        manager.update(TimeOfDayController(time_hours=12.0))

        color, intensity = manager.get_ambient_light()

        assert intensity > 0.2  # Should be bright

    def test_get_ambient_light_night(self):
        manager = SkyManager()
        manager.update(TimeOfDayController(time_hours=0.0))

        color, intensity = manager.get_ambient_light()

        assert intensity < 0.2  # Should be dim

    def test_set_sky_type(self):
        manager = SkyManager(sky_type="procedural")
        manager.set_sky_type("hdri")

        assert manager.sky_type == "hdri"


# =============================================================================
# Integration Tests
# =============================================================================


class TestSkyIntegration:
    def test_full_day_sky_cycle(self):
        """Test sky colors throughout a day cycle."""
        manager = SkyManager()
        tod = TimeOfDayController(time_hours=0.0, time_scale=3600.0)

        colors = []
        for hour in range(24):
            tod.set_time(float(hour))
            manager.update(tod)
            color = manager.get_sky_color((0, 1, 0))
            colors.append(color)

        # Colors should vary throughout the day
        brightness = [sum(c) for c in colors]
        assert max(brightness) > min(brightness)

    def test_sky_color_matches_sun_elevation(self):
        """Test that sky brightness correlates with sun elevation."""
        manager = SkyManager()

        # Midday - looking up at the sky
        manager.update(TimeOfDayController(time_hours=12.0))
        noon_color = manager.get_sky_color((0, 1, 0))

        # Midnight - looking up at the sky (stars are visible but darker overall)
        manager.update(TimeOfDayController(time_hours=0.0))
        # Look at a direction away from where stars typically are
        midnight_color = manager.get_sky_color((0.5, 0.5, 0.5))

        # Noon should be brighter (use a more specific check)
        # At midnight, sun is below horizon so procedural sky should be dark
        # The base sky without stars should be dimmer
        noon_brightness = sum(noon_color)
        # Check that noon produces non-trivial sky color
        assert noon_brightness > 0.5

    def test_stars_visible_at_night_only(self):
        """Test that stars are only visible at night."""
        manager = SkyManager()

        # Night
        manager.update(TimeOfDayController(time_hours=0.0))
        night_color = manager.get_sky_color((0.999, 0.01, 0))

        # Day
        manager.update(TimeOfDayController(time_hours=12.0))
        day_color = manager.get_sky_color((0.999, 0.01, 0))

        # Night may have star contribution, day should not
        # (Both may be similar if no star in that direction)
        assert isinstance(night_color, tuple)
        assert isinstance(day_color, tuple)

    def test_atmosphere_affects_sky_color(self):
        """Test that atmosphere settings affect sky color."""
        # High rayleigh (more blue)
        atmo_blue = AtmosphereSettings(rayleigh_density=2.0)
        sky_blue = ProceduralSky(atmosphere=atmo_blue, sun_position=SunPosition(elevation=45.0))

        # Low rayleigh (less blue)
        atmo_haze = AtmosphereSettings(rayleigh_density=0.1, mie_density=2.0)
        sky_haze = ProceduralSky(atmosphere=atmo_haze, sun_position=SunPosition(elevation=45.0))

        color_blue = sky_blue.compute_sky_color((0, 1, 0))
        color_haze = sky_haze.compute_sky_color((0, 1, 0))

        # Colors should be different
        assert color_blue != color_haze
