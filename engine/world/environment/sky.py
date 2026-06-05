"""
Sky System - Procedural and HDRI sky rendering.

Provides atmosphere simulation using Rayleigh and Mie scattering,
as well as HDRI and static skybox support. Includes celestial body
management for sun, moon, and stars.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from engine.world.environment.constants import (
    DEFAULT_ATMOSPHERE_HEIGHT,
    EARTH_RADIUS,
    SUN_ANGULAR_RADIUS,
    DEFAULT_RAYLEIGH_DENSITY,
    DEFAULT_MIE_DENSITY,
    DEFAULT_MIE_ANISOTROPY,
    DEFAULT_OZONE_DENSITY,
    DEFAULT_RAYLEIGH_COLOR,
    DEFAULT_MIE_COLOR,
    DEFAULT_OZONE_COLOR,
)
from engine.world.environment.time_of_day import SunPosition, TimeOfDayController

# Type aliases
Color3 = Tuple[float, float, float]
Direction3 = Tuple[float, float, float]


@dataclass
class AtmosphereSettings:
    """
    Settings for atmospheric scattering simulation.

    Based on Nishita's atmospheric scattering model with
    Rayleigh (blue sky), Mie (sun glow/haze), and ozone absorption.
    """
    # Rayleigh scattering (responsible for blue sky)
    rayleigh_density: float = 1.0
    rayleigh_color: Color3 = (0.175, 0.409, 1.0)  # Blue wavelength scattering

    # Mie scattering (responsible for sun glow and haze)
    mie_density: float = 1.0
    mie_color: Color3 = (1.0, 1.0, 1.0)
    mie_anisotropy: float = 0.8  # Forward scattering factor (0-1)

    # Ozone absorption
    ozone_density: float = 1.0
    ozone_color: Color3 = (0.65, 1.0, 0.85)  # Absorption color

    # Scale
    atmosphere_height: float = 100000.0  # meters
    planet_radius: float = 6371000.0  # Earth radius in meters

    # Sun
    sun_angular_radius: float = 0.00465  # radians (~0.27 degrees)
    sun_intensity_multiplier: float = 1.0

    # Ground
    ground_albedo: Color3 = (0.3, 0.3, 0.3)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rayleigh_density": self.rayleigh_density,
            "rayleigh_color": self.rayleigh_color,
            "mie_density": self.mie_density,
            "mie_color": self.mie_color,
            "mie_anisotropy": self.mie_anisotropy,
            "ozone_density": self.ozone_density,
            "ozone_color": self.ozone_color,
            "atmosphere_height": self.atmosphere_height,
            "planet_radius": self.planet_radius,
            "sun_angular_radius": self.sun_angular_radius,
            "sun_intensity_multiplier": self.sun_intensity_multiplier,
            "ground_albedo": self.ground_albedo,
        }

    def lerp(self, other: "AtmosphereSettings", t: float) -> "AtmosphereSettings":
        """Interpolate between atmosphere settings."""
        t = max(0.0, min(1.0, t))
        inv_t = 1.0 - t

        def lerp_color(a: Color3, b: Color3) -> Color3:
            return (
                a[0] * inv_t + b[0] * t,
                a[1] * inv_t + b[1] * t,
                a[2] * inv_t + b[2] * t,
            )

        return AtmosphereSettings(
            rayleigh_density=self.rayleigh_density * inv_t + other.rayleigh_density * t,
            rayleigh_color=lerp_color(self.rayleigh_color, other.rayleigh_color),
            mie_density=self.mie_density * inv_t + other.mie_density * t,
            mie_color=lerp_color(self.mie_color, other.mie_color),
            mie_anisotropy=self.mie_anisotropy * inv_t + other.mie_anisotropy * t,
            ozone_density=self.ozone_density * inv_t + other.ozone_density * t,
            ozone_color=lerp_color(self.ozone_color, other.ozone_color),
            atmosphere_height=self.atmosphere_height * inv_t + other.atmosphere_height * t,
            planet_radius=self.planet_radius * inv_t + other.planet_radius * t,
            sun_angular_radius=self.sun_angular_radius * inv_t + other.sun_angular_radius * t,
            sun_intensity_multiplier=self.sun_intensity_multiplier * inv_t + other.sun_intensity_multiplier * t,
            ground_albedo=lerp_color(self.ground_albedo, other.ground_albedo),
        )


class BaseSky(ABC):
    """Abstract base class for sky implementations."""

    @abstractmethod
    def get_sky_color(self, direction: Direction3) -> Color3:
        """
        Get sky color in a direction.

        Args:
            direction: Normalized view direction (x, y, z).

        Returns:
            RGB color (0-1 range).
        """
        pass

    @abstractmethod
    def update(self, time_of_day: float) -> None:
        """
        Update sky state based on time of day.

        Args:
            time_of_day: Time in hours (0-24).
        """
        pass


class ProceduralSky(BaseSky):
    """
    Procedural sky using atmospheric scattering.

    Simulates realistic sky colors based on sun position
    using Rayleigh and Mie scattering approximations.
    """

    def __init__(
        self,
        atmosphere: Optional[AtmosphereSettings] = None,
        sun_position: Optional[SunPosition] = None,
    ) -> None:
        """
        Initialize procedural sky.

        Args:
            atmosphere: Atmosphere settings.
            sun_position: Initial sun position.
        """
        self.atmosphere = atmosphere or AtmosphereSettings()
        self.sun_position = sun_position or SunPosition(azimuth=180.0, elevation=45.0)

    def compute_sky_color(self, view_direction: Direction3) -> Color3:
        """
        Compute sky color for a view direction.

        Uses simplified Rayleigh/Mie scattering approximation.

        Args:
            view_direction: Normalized view direction.

        Returns:
            RGB color (HDR, may exceed 1.0).
        """
        # Get sun direction
        sun_dir = self.sun_position.direction

        # Calculate view-sun angle
        dot = (
            view_direction[0] * sun_dir[0] +
            view_direction[1] * sun_dir[1] +
            view_direction[2] * sun_dir[2]
        )
        dot = max(-1.0, min(1.0, dot))
        sun_angle = math.acos(dot)

        # Rayleigh scattering (blue sky)
        # Stronger when looking away from sun, varies with view elevation
        view_elevation = max(0.0, view_direction[1])  # 0 at horizon, 1 at zenith

        rayleigh_factor = 1.0 - abs(dot)
        rayleigh_r = self.atmosphere.rayleigh_color[0] * rayleigh_factor * self.atmosphere.rayleigh_density
        rayleigh_g = self.atmosphere.rayleigh_color[1] * rayleigh_factor * self.atmosphere.rayleigh_density
        rayleigh_b = self.atmosphere.rayleigh_color[2] * rayleigh_factor * self.atmosphere.rayleigh_density

        # Mie scattering (sun glow)
        # Uses Henyey-Greenstein phase function approximation
        g = self.atmosphere.mie_anisotropy
        mie_phase = (1 - g * g) / (4 * math.pi * pow(1 + g * g - 2 * g * dot, 1.5) + 0.001)
        mie_factor = mie_phase * self.atmosphere.mie_density * 0.1

        mie_r = self.atmosphere.mie_color[0] * mie_factor
        mie_g = self.atmosphere.mie_color[1] * mie_factor
        mie_b = self.atmosphere.mie_color[2] * mie_factor

        # Sun altitude factor (affects overall brightness)
        sun_elevation = self.sun_position.elevation
        sun_factor = max(0.0, (sun_elevation + 10) / 100.0)  # Soft transition below horizon

        # Combine scattering contributions
        sky_r = (rayleigh_r * 0.5 + mie_r) * sun_factor * self.atmosphere.sun_intensity_multiplier
        sky_g = (rayleigh_g * 0.7 + mie_g) * sun_factor * self.atmosphere.sun_intensity_multiplier
        sky_b = (rayleigh_b * 1.0 + mie_b) * sun_factor * self.atmosphere.sun_intensity_multiplier

        # Add horizon glow during sunrise/sunset
        if 0 < sun_elevation < 15:
            horizon_factor = (1 - view_elevation) * (1 - sun_elevation / 15)
            sky_r += horizon_factor * 0.8
            sky_g += horizon_factor * 0.4
            sky_b += horizon_factor * 0.2

        # Clamp to reasonable HDR range
        return (
            max(0.0, min(10.0, sky_r)),
            max(0.0, min(10.0, sky_g)),
            max(0.0, min(10.0, sky_b)),
        )

    def compute_sun_disk(self, view_direction: Direction3) -> float:
        """
        Compute sun disk intensity for a view direction.

        Args:
            view_direction: Normalized view direction.

        Returns:
            Sun disk intensity (0 = not visible, >0 = visible).
        """
        if self.sun_position.elevation < -5:
            return 0.0  # Sun below horizon

        sun_dir = self.sun_position.direction

        # Calculate angle to sun
        dot = (
            view_direction[0] * sun_dir[0] +
            view_direction[1] * sun_dir[1] +
            view_direction[2] * sun_dir[2]
        )

        angle = math.acos(max(-1.0, min(1.0, dot)))

        # Sun disk
        if angle < self.atmosphere.sun_angular_radius:
            # Inside sun disk - full brightness
            return self.atmosphere.sun_intensity_multiplier * 100.0
        elif angle < self.atmosphere.sun_angular_radius * 2:
            # Corona/glow falloff
            t = (angle - self.atmosphere.sun_angular_radius) / self.atmosphere.sun_angular_radius
            return self.atmosphere.sun_intensity_multiplier * 100.0 * (1 - t)

        return 0.0

    def get_aerial_perspective(self, distance: float) -> float:
        """
        Calculate aerial perspective fog factor.

        Args:
            distance: Distance from camera in meters.

        Returns:
            Fog factor (0 = no fog, 1 = full fog).
        """
        # Exponential falloff based on atmosphere density
        density = (self.atmosphere.rayleigh_density + self.atmosphere.mie_density) * 0.00001
        return 1.0 - math.exp(-distance * density)

    def get_sky_color(self, direction: Direction3) -> Color3:
        """Get sky color (interface implementation)."""
        return self.compute_sky_color(direction)

    def update(self, time_of_day: float) -> None:
        """Update sun position based on time (simplified)."""
        # Basic sun position update (real implementation would use TODController)
        hour_angle = (time_of_day - 12.0) * 15.0
        elevation = 90.0 - abs(hour_angle) * 2  # Simplified parabolic arc
        self.sun_position = SunPosition(
            azimuth=(180.0 + hour_angle) % 360.0,
            elevation=max(-90.0, min(90.0, elevation)),
        )


class HDRISky(BaseSky):
    """
    HDRI (High Dynamic Range Image) based sky.

    Uses a pre-captured environment map for sky rendering.
    """

    def __init__(
        self,
        texture_path: str = "",
        rotation: float = 0.0,
        intensity: float = 1.0,
    ) -> None:
        """
        Initialize HDRI sky.

        Args:
            texture_path: Path to HDRI texture.
            rotation: Rotation around Y axis in degrees.
            intensity: Brightness multiplier.
        """
        self.texture_path = texture_path
        self.rotation = rotation
        self.intensity = intensity
        self._loaded = False

        # Placeholder for texture data (would be loaded from file)
        self._default_color: Color3 = (0.5, 0.6, 0.8)

    def load(self) -> bool:
        """
        Load HDRI texture.

        Returns:
            True if loaded successfully.
        """
        # In a real implementation, this would load the actual HDRI
        self._loaded = bool(self.texture_path)
        return self._loaded

    def get_sky_color(self, direction: Direction3) -> Color3:
        """Get sky color from HDRI (simplified)."""
        # Apply rotation
        rot_rad = math.radians(self.rotation)
        cos_r = math.cos(rot_rad)
        sin_r = math.sin(rot_rad)

        rotated_dir = (
            direction[0] * cos_r - direction[2] * sin_r,
            direction[1],
            direction[0] * sin_r + direction[2] * cos_r,
        )

        # In a real implementation, sample from the HDRI texture
        # For now, return a gradient based on direction
        elevation = math.asin(max(-1, min(1, rotated_dir[1])))
        elevation_factor = (elevation + math.pi/2) / math.pi  # 0 at bottom, 1 at top

        color = (
            self._default_color[0] * elevation_factor * self.intensity,
            self._default_color[1] * elevation_factor * self.intensity,
            self._default_color[2] * elevation_factor * self.intensity,
        )
        return color

    def update(self, time_of_day: float) -> None:
        """HDRI sky doesn't update with time (static)."""
        pass


class StaticSky(BaseSky):
    """
    Static skybox using 6 face textures.

    Traditional cubemap-based sky rendering.
    """

    def __init__(
        self,
        skybox_textures: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize static skybox.

        Args:
            skybox_textures: List of 6 texture paths
                             [+X, -X, +Y, -Y, +Z, -Z].
        """
        self.skybox_textures = skybox_textures or []
        self._loaded = False

        # Default colors for each face
        self._face_colors: List[Color3] = [
            (0.6, 0.7, 0.9),  # +X
            (0.6, 0.7, 0.9),  # -X
            (0.4, 0.5, 0.9),  # +Y (top)
            (0.3, 0.35, 0.4),  # -Y (bottom)
            (0.6, 0.7, 0.9),  # +Z
            (0.6, 0.7, 0.9),  # -Z
        ]

    def get_sky_color(self, direction: Direction3) -> Color3:
        """Get sky color from skybox."""
        # Determine dominant face
        abs_x = abs(direction[0])
        abs_y = abs(direction[1])
        abs_z = abs(direction[2])

        if abs_x >= abs_y and abs_x >= abs_z:
            face_index = 0 if direction[0] > 0 else 1
        elif abs_y >= abs_x and abs_y >= abs_z:
            face_index = 2 if direction[1] > 0 else 3
        else:
            face_index = 4 if direction[2] > 0 else 5

        return self._face_colors[face_index]

    def update(self, time_of_day: float) -> None:
        """Static skybox doesn't update with time."""
        pass


class CelestialBodyType(Enum):
    """Types of celestial bodies."""
    SUN = auto()
    MOON = auto()
    PLANET = auto()
    STAR = auto()


@dataclass
class CelestialBody:
    """
    A celestial body in the sky (sun, moon, planet, etc.).
    """
    body_type: CelestialBodyType
    angular_size: float = 0.5  # degrees
    color: Color3 = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    texture_path: Optional[str] = None

    # Position calculation parameters
    orbital_period: float = 24.0  # hours for a full cycle
    orbital_offset: float = 0.0  # phase offset in hours
    orbital_inclination: float = 0.0  # degrees from ecliptic

    # For moon: phase
    _current_phase: float = 0.5  # 0 = new, 0.5 = full, 1 = new again

    def get_position(self, time_of_day: float, latitude: float = 45.0) -> Direction3:
        """
        Calculate celestial body position.

        Args:
            time_of_day: Time in hours (0-24).
            latitude: Observer latitude.

        Returns:
            Normalized direction vector.
        """
        # Calculate hour angle (position in sky)
        phase = ((time_of_day + self.orbital_offset) % self.orbital_period) / self.orbital_period
        hour_angle = phase * 2 * math.pi - math.pi  # -pi to +pi

        # Calculate declination (seasonal variation)
        inclination_rad = math.radians(self.orbital_inclination)
        declination = math.sin(hour_angle * 0.1) * inclination_rad  # Simplified

        # Calculate elevation and azimuth
        lat_rad = math.radians(latitude)

        sin_elevation = (
            math.sin(lat_rad) * math.sin(declination) +
            math.cos(lat_rad) * math.cos(declination) * math.cos(hour_angle)
        )
        elevation = math.asin(max(-1, min(1, sin_elevation)))

        cos_azimuth = (
            (math.sin(declination) - math.sin(lat_rad) * sin_elevation) /
            (math.cos(lat_rad) * math.cos(elevation) + 0.001)
        )
        azimuth = math.acos(max(-1, min(1, cos_azimuth)))

        if hour_angle > 0:
            azimuth = 2 * math.pi - azimuth

        # Convert to direction vector
        x = math.sin(azimuth) * math.cos(elevation)
        y = math.sin(elevation)
        z = math.cos(azimuth) * math.cos(elevation)

        return (x, y, z)

    def get_visible_intensity(self, direction: Direction3, time_of_day: float) -> float:
        """
        Get visible intensity based on position and time.

        Args:
            direction: Direction to check.
            time_of_day: Current time.

        Returns:
            Intensity multiplier (0 = not visible).
        """
        body_pos = self.get_position(time_of_day)

        # Check if below horizon
        if body_pos[1] < -0.1:
            return 0.0

        # Calculate angle to body
        dot = (
            direction[0] * body_pos[0] +
            direction[1] * body_pos[1] +
            direction[2] * body_pos[2]
        )

        angle = math.acos(max(-1, min(1, dot)))
        angular_radius = math.radians(self.angular_size / 2)

        if angle < angular_radius:
            return self.intensity
        elif angle < angular_radius * 3:
            # Corona/glow
            t = (angle - angular_radius) / (angular_radius * 2)
            return self.intensity * (1 - t) * 0.3

        return 0.0

    def get_moon_phase(self) -> float:
        """Get current moon phase (0-1)."""
        return self._current_phase

    def set_moon_phase(self, phase: float) -> None:
        """Set moon phase (0-1)."""
        self._current_phase = phase % 1.0


class StarField:
    """
    Procedural star field for night sky.
    """

    def __init__(
        self,
        star_count: int = 5000,
        brightness: float = 1.0,
        twinkle_speed: float = 1.0,
    ) -> None:
        """
        Initialize star field.

        Args:
            star_count: Number of stars.
            brightness: Overall brightness.
            twinkle_speed: Speed of star twinkling.
        """
        self.star_count = star_count
        self.brightness = brightness
        self.twinkle_speed = twinkle_speed
        self._stars: List[Tuple[Direction3, float, Color3]] = []
        self._generate_stars()

    def _generate_stars(self) -> None:
        """Generate random star positions and properties."""
        import random

        self._stars = []
        for _ in range(self.star_count):
            # Random direction on unit sphere
            theta = random.uniform(0, 2 * math.pi)
            phi = math.acos(random.uniform(-1, 1))

            direction = (
                math.sin(phi) * math.cos(theta),
                abs(math.sin(phi) * math.sin(theta)),  # Only upper hemisphere
                math.cos(phi),
            )

            # Random brightness
            brightness = random.uniform(0.1, 1.0) ** 2  # More dim stars

            # Random color (slight variations)
            color_temp = random.uniform(0.8, 1.2)
            color = (
                min(1.0, 0.9 * color_temp),
                min(1.0, 0.95 * color_temp),
                min(1.0, 1.0 * color_temp),
            )

            self._stars.append((direction, brightness, color))

    def get_star_brightness(
        self,
        direction: Direction3,
        time: float,
        sun_elevation: float,
    ) -> Color3:
        """
        Get star contribution at a direction.

        Args:
            direction: View direction.
            time: Time for twinkling.
            sun_elevation: Sun elevation (stars fade during day).

        Returns:
            RGB color contribution.
        """
        # Stars only visible at night
        if sun_elevation > 5:
            return (0.0, 0.0, 0.0)

        # Fade stars as sun rises
        visibility = 1.0
        if sun_elevation > -10:
            visibility = (-sun_elevation + 5) / 15

        total_r, total_g, total_b = 0.0, 0.0, 0.0

        for star_dir, brightness, color in self._stars:
            # Check if star is in view direction (with tolerance)
            dot = (
                direction[0] * star_dir[0] +
                direction[1] * star_dir[1] +
                direction[2] * star_dir[2]
            )

            if dot > 0.999:  # Very close to star
                # Add twinkling
                twinkle = 0.8 + 0.2 * math.sin(
                    time * self.twinkle_speed + star_dir[0] * 1000
                )

                contribution = brightness * self.brightness * visibility * twinkle
                total_r += color[0] * contribution
                total_g += color[1] * contribution
                total_b += color[2] * contribution

        return (total_r, total_g, total_b)


class SkyManager:
    """
    Manages sky rendering including atmosphere, celestial bodies, and stars.
    """

    def __init__(
        self,
        sky_type: str = "procedural",
        atmosphere: Optional[AtmosphereSettings] = None,
    ) -> None:
        """
        Initialize sky manager.

        Args:
            sky_type: Type of sky ("procedural", "hdri", "static").
            atmosphere: Atmosphere settings for procedural sky.
        """
        self.sky_type = sky_type
        self.atmosphere = atmosphere or AtmosphereSettings()

        # Sky implementation
        if sky_type == "procedural":
            self._sky: BaseSky = ProceduralSky(atmosphere=self.atmosphere)
        elif sky_type == "hdri":
            self._sky = HDRISky()
        else:
            self._sky = StaticSky()

        # Celestial bodies
        self.sun = CelestialBody(
            body_type=CelestialBodyType.SUN,
            angular_size=0.53,
            color=(1.0, 0.95, 0.9),
            intensity=100.0,
            orbital_period=24.0,
        )

        self.moon = CelestialBody(
            body_type=CelestialBodyType.MOON,
            angular_size=0.52,
            color=(0.9, 0.95, 1.0),
            intensity=0.2,
            orbital_period=24.8,  # Moon rises ~50 min later each day
            orbital_offset=12.0,  # Offset from sun
        )

        self.stars = StarField(star_count=3000)

        # Current state
        self._current_time: float = 12.0
        self._sun_position: SunPosition = SunPosition()

    def update(self, tod_controller: Optional[TimeOfDayController] = None) -> None:
        """
        Update sky state.

        Args:
            tod_controller: Time of day controller for sun position.
        """
        if tod_controller:
            self._current_time = tod_controller.time_hours
            self._sun_position = tod_controller.get_sun_position()

            # Update procedural sky
            if isinstance(self._sky, ProceduralSky):
                self._sky.sun_position = self._sun_position

        self._sky.update(self._current_time)

    def get_sky_color(self, direction: Direction3) -> Color3:
        """
        Get complete sky color including atmosphere and celestial bodies.

        Args:
            direction: Normalized view direction.

        Returns:
            RGB color (HDR).
        """
        # Base sky color
        color = self._sky.get_sky_color(direction)

        # Add stars at night
        if self._sun_position.elevation < 5:
            stars = self.stars.get_star_brightness(
                direction, self._current_time, self._sun_position.elevation
            )
            color = (
                color[0] + stars[0],
                color[1] + stars[1],
                color[2] + stars[2],
            )

        return color

    def get_sun_position(self) -> SunPosition:
        """Get current sun position."""
        return self._sun_position

    def get_ambient_light(self) -> Tuple[Color3, float]:
        """
        Get ambient sky light color and intensity.

        Returns:
            Tuple of (color, intensity).
        """
        elevation = self._sun_position.elevation

        if elevation > 10:
            # Daytime
            return ((0.5, 0.6, 0.8), 0.4)
        elif elevation > -5:
            # Sunrise/sunset
            t = (elevation + 5) / 15
            color = (
                0.3 + 0.2 * t,
                0.2 + 0.4 * t,
                0.3 + 0.5 * t,
            )
            return (color, 0.2 + 0.2 * t)
        else:
            # Night
            return ((0.1, 0.1, 0.2), 0.1)

    def set_sky_type(self, sky_type: str) -> None:
        """Change sky type."""
        self.sky_type = sky_type
        if sky_type == "procedural":
            self._sky = ProceduralSky(atmosphere=self.atmosphere)
        elif sky_type == "hdri":
            self._sky = HDRISky()
        else:
            self._sky = StaticSky()


__all__ = [
    # Data classes
    "AtmosphereSettings",
    # Enums
    "CelestialBodyType",
    # Sky implementations
    "BaseSky",
    "ProceduralSky",
    "HDRISky",
    "StaticSky",
    # Celestial
    "CelestialBody",
    "StarField",
    # Manager
    "SkyManager",
    # Types
    "Color3",
    "Direction3",
]
