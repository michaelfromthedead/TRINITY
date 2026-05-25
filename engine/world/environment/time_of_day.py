"""
Time of Day System - Astronomical time simulation with lighting.

Provides accurate sun position calculation based on time and latitude,
with smooth interpolation between lighting keyframes for different
periods of the day.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

# Type aliases
Color3 = Tuple[float, float, float]
TimeCallback = Callable[["TimeOfDayPeriod", "TimeOfDayPeriod"], None]


class TimeOfDayPeriod(Enum):
    """Periods of the day."""
    NIGHT = auto()       # 0:00 - 4:30
    DAWN = auto()        # 4:30 - 6:00
    SUNRISE = auto()     # 6:00 - 7:30
    MORNING = auto()     # 7:30 - 11:00
    NOON = auto()        # 11:00 - 13:00
    AFTERNOON = auto()   # 13:00 - 17:00
    SUNSET = auto()      # 17:00 - 19:30
    DUSK = auto()        # 19:30 - 21:00


# Period boundaries (hours)
PERIOD_BOUNDARIES: Dict[TimeOfDayPeriod, Tuple[float, float]] = {
    TimeOfDayPeriod.NIGHT: (0.0, 4.5),
    TimeOfDayPeriod.DAWN: (4.5, 6.0),
    TimeOfDayPeriod.SUNRISE: (6.0, 7.5),
    TimeOfDayPeriod.MORNING: (7.5, 11.0),
    TimeOfDayPeriod.NOON: (11.0, 13.0),
    TimeOfDayPeriod.AFTERNOON: (13.0, 17.0),
    TimeOfDayPeriod.SUNSET: (17.0, 19.5),
    TimeOfDayPeriod.DUSK: (19.5, 21.0),
}


@dataclass
class SunPosition:
    """
    Sun position in the sky.

    Uses spherical coordinates with azimuth (compass direction)
    and elevation (angle above horizon).
    """
    azimuth: float = 180.0  # Degrees, 0 = North, 90 = East, 180 = South
    elevation: float = 45.0  # Degrees, 0 = horizon, 90 = zenith, negative = below horizon

    @property
    def direction(self) -> Tuple[float, float, float]:
        """
        Get sun direction vector (pointing toward sun).

        Returns:
            Normalized direction vector (x, y, z).
        """
        # Convert to radians
        az_rad = math.radians(self.azimuth)
        el_rad = math.radians(self.elevation)

        # Calculate direction (y is up)
        x = math.sin(az_rad) * math.cos(el_rad)
        y = math.sin(el_rad)
        z = math.cos(az_rad) * math.cos(el_rad)

        return (x, y, z)

    @property
    def is_day(self) -> bool:
        """Check if sun is above horizon."""
        return self.elevation > 0

    @property
    def is_golden_hour(self) -> bool:
        """Check if sun is in golden hour (low angle, warm light)."""
        return 0 < self.elevation < 15

    def lerp(self, other: "SunPosition", t: float) -> "SunPosition":
        """
        Interpolate between two sun positions.

        Args:
            other: Target position.
            t: Interpolation factor (0-1).

        Returns:
            Interpolated position.
        """
        t = max(0.0, min(1.0, t))

        # Interpolate azimuth taking shortest path
        az_diff = ((other.azimuth - self.azimuth + 180) % 360) - 180
        new_azimuth = (self.azimuth + az_diff * t) % 360

        # Linear interpolation for elevation
        new_elevation = self.elevation + (other.elevation - self.elevation) * t

        return SunPosition(azimuth=new_azimuth, elevation=new_elevation)


@dataclass
class TODLighting:
    """
    Lighting configuration for a time of day.

    Includes sun/moon properties and ambient lighting.
    """
    # Sun properties
    sun_color: Color3 = (1.0, 0.95, 0.9)
    sun_intensity: float = 1.0

    # Ambient lighting
    ambient_color: Color3 = (0.5, 0.6, 0.7)
    ambient_intensity: float = 0.3

    # Sky colors (for atmosphere)
    sky_color_zenith: Color3 = (0.3, 0.5, 0.9)
    sky_color_horizon: Color3 = (0.7, 0.8, 0.9)

    # Fog (atmosphere)
    fog_color: Color3 = (0.7, 0.8, 0.9)
    fog_density: float = 0.0

    # Moon (for night)
    moon_color: Color3 = (0.6, 0.7, 0.8)
    moon_intensity: float = 0.0

    # Shadows
    shadow_strength: float = 1.0
    shadow_softness: float = 0.5

    def lerp(self, other: "TODLighting", t: float) -> "TODLighting":
        """
        Interpolate between two lighting configurations.

        Args:
            other: Target lighting.
            t: Interpolation factor (0-1).

        Returns:
            Interpolated lighting.
        """
        t = max(0.0, min(1.0, t))
        inv_t = 1.0 - t

        def lerp_color(a: Color3, b: Color3) -> Color3:
            return (
                a[0] * inv_t + b[0] * t,
                a[1] * inv_t + b[1] * t,
                a[2] * inv_t + b[2] * t,
            )

        return TODLighting(
            sun_color=lerp_color(self.sun_color, other.sun_color),
            sun_intensity=self.sun_intensity * inv_t + other.sun_intensity * t,
            ambient_color=lerp_color(self.ambient_color, other.ambient_color),
            ambient_intensity=self.ambient_intensity * inv_t + other.ambient_intensity * t,
            sky_color_zenith=lerp_color(self.sky_color_zenith, other.sky_color_zenith),
            sky_color_horizon=lerp_color(self.sky_color_horizon, other.sky_color_horizon),
            fog_color=lerp_color(self.fog_color, other.fog_color),
            fog_density=self.fog_density * inv_t + other.fog_density * t,
            moon_color=lerp_color(self.moon_color, other.moon_color),
            moon_intensity=self.moon_intensity * inv_t + other.moon_intensity * t,
            shadow_strength=self.shadow_strength * inv_t + other.shadow_strength * t,
            shadow_softness=self.shadow_softness * inv_t + other.shadow_softness * t,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sun_color": self.sun_color,
            "sun_intensity": self.sun_intensity,
            "ambient_color": self.ambient_color,
            "ambient_intensity": self.ambient_intensity,
            "sky_color_zenith": self.sky_color_zenith,
            "sky_color_horizon": self.sky_color_horizon,
            "fog_color": self.fog_color,
            "fog_density": self.fog_density,
            "moon_color": self.moon_color,
            "moon_intensity": self.moon_intensity,
            "shadow_strength": self.shadow_strength,
            "shadow_softness": self.shadow_softness,
        }


@dataclass
class TODKeyframe:
    """
    A keyframe in the time-of-day lighting curve.

    Defines lighting at a specific time of day.
    """
    time_hours: float  # 0-24
    lighting: TODLighting

    def __post_init__(self) -> None:
        # Normalize time to 0-24 range
        self.time_hours = self.time_hours % 24.0


class TODCurve:
    """
    Curve of lighting keyframes for a full day cycle.

    Supports interpolation between keyframes.
    """

    def __init__(self, keyframes: Optional[List[TODKeyframe]] = None) -> None:
        """
        Initialize TOD curve.

        Args:
            keyframes: List of keyframes (should be sorted by time).
        """
        self._keyframes: List[TODKeyframe] = []
        if keyframes:
            for kf in keyframes:
                self.add_keyframe(kf)

    def add_keyframe(self, keyframe: TODKeyframe) -> None:
        """Add a keyframe to the curve."""
        # Insert in sorted order
        for i, existing in enumerate(self._keyframes):
            if keyframe.time_hours < existing.time_hours:
                self._keyframes.insert(i, keyframe)
                return
        self._keyframes.append(keyframe)

    def remove_keyframe(self, time_hours: float) -> Optional[TODKeyframe]:
        """Remove keyframe at a specific time."""
        for i, kf in enumerate(self._keyframes):
            if abs(kf.time_hours - time_hours) < 0.01:
                return self._keyframes.pop(i)
        return None

    def interpolate(self, time_hours: float) -> TODLighting:
        """
        Interpolate lighting at a specific time.

        Args:
            time_hours: Time in hours (0-24).

        Returns:
            Interpolated lighting configuration.
        """
        if not self._keyframes:
            return TODLighting()

        if len(self._keyframes) == 1:
            return self._keyframes[0].lighting

        time_hours = time_hours % 24.0

        # Find surrounding keyframes
        prev_kf: Optional[TODKeyframe] = None
        next_kf: Optional[TODKeyframe] = None

        for i, kf in enumerate(self._keyframes):
            if kf.time_hours <= time_hours:
                prev_kf = kf
            if kf.time_hours > time_hours and next_kf is None:
                next_kf = kf
                break

        # Handle wrap-around
        if prev_kf is None:
            prev_kf = self._keyframes[-1]
        if next_kf is None:
            next_kf = self._keyframes[0]

        # Calculate interpolation factor
        if prev_kf.time_hours == next_kf.time_hours:
            return prev_kf.lighting

        # Handle wrap-around for time difference
        prev_time = prev_kf.time_hours
        next_time = next_kf.time_hours

        if next_time < prev_time:
            # Wrap around midnight
            next_time += 24.0
            if time_hours < prev_time:
                time_hours += 24.0

        duration = next_time - prev_time
        elapsed = time_hours - prev_time
        t = elapsed / duration if duration > 0 else 0

        # Use smoothstep for smoother transitions
        t = t * t * (3 - 2 * t)

        return prev_kf.lighting.lerp(next_kf.lighting, t)

    @property
    def keyframes(self) -> List[TODKeyframe]:
        """Get all keyframes."""
        return list(self._keyframes)


class TimeOfDayPreset:
    """
    Predefined time-of-day configurations.

    Contains a full curve of keyframes for common presets.
    """

    @staticmethod
    def realistic() -> TODCurve:
        """Create a realistic day/night cycle."""
        return TODCurve([
            # Night (0:00)
            TODKeyframe(0.0, TODLighting(
                sun_color=(0.0, 0.0, 0.0),
                sun_intensity=0.0,
                ambient_color=(0.05, 0.05, 0.1),
                ambient_intensity=0.1,
                sky_color_zenith=(0.02, 0.02, 0.05),
                sky_color_horizon=(0.05, 0.05, 0.1),
                fog_color=(0.02, 0.02, 0.05),
                fog_density=0.05,
                moon_color=(0.6, 0.7, 0.8),
                moon_intensity=0.15,
                shadow_strength=0.3,
            )),
            # Dawn (5:00)
            TODKeyframe(5.0, TODLighting(
                sun_color=(1.0, 0.6, 0.3),
                sun_intensity=0.1,
                ambient_color=(0.2, 0.15, 0.2),
                ambient_intensity=0.2,
                sky_color_zenith=(0.2, 0.15, 0.3),
                sky_color_horizon=(0.9, 0.5, 0.3),
                fog_color=(0.5, 0.3, 0.2),
                fog_density=0.1,
                moon_intensity=0.05,
                shadow_strength=0.5,
            )),
            # Sunrise (6:30)
            TODKeyframe(6.5, TODLighting(
                sun_color=(1.0, 0.8, 0.5),
                sun_intensity=0.5,
                ambient_color=(0.4, 0.3, 0.3),
                ambient_intensity=0.3,
                sky_color_zenith=(0.4, 0.5, 0.8),
                sky_color_horizon=(1.0, 0.7, 0.4),
                fog_color=(0.8, 0.6, 0.4),
                fog_density=0.05,
                moon_intensity=0.0,
                shadow_strength=0.7,
            )),
            # Morning (9:00)
            TODKeyframe(9.0, TODLighting(
                sun_color=(1.0, 0.95, 0.85),
                sun_intensity=0.85,
                ambient_color=(0.5, 0.55, 0.6),
                ambient_intensity=0.35,
                sky_color_zenith=(0.35, 0.55, 0.95),
                sky_color_horizon=(0.7, 0.8, 0.95),
                fog_color=(0.7, 0.8, 0.9),
                fog_density=0.02,
                shadow_strength=0.9,
            )),
            # Noon (12:00)
            TODKeyframe(12.0, TODLighting(
                sun_color=(1.0, 0.98, 0.95),
                sun_intensity=1.0,
                ambient_color=(0.55, 0.6, 0.7),
                ambient_intensity=0.4,
                sky_color_zenith=(0.3, 0.5, 0.95),
                sky_color_horizon=(0.6, 0.75, 0.95),
                fog_color=(0.7, 0.8, 0.95),
                fog_density=0.01,
                shadow_strength=1.0,
                shadow_softness=0.3,
            )),
            # Afternoon (15:00)
            TODKeyframe(15.0, TODLighting(
                sun_color=(1.0, 0.95, 0.88),
                sun_intensity=0.9,
                ambient_color=(0.5, 0.55, 0.65),
                ambient_intensity=0.35,
                sky_color_zenith=(0.32, 0.52, 0.92),
                sky_color_horizon=(0.65, 0.78, 0.92),
                fog_color=(0.7, 0.8, 0.92),
                fog_density=0.015,
                shadow_strength=0.95,
            )),
            # Sunset (18:30)
            TODKeyframe(18.5, TODLighting(
                sun_color=(1.0, 0.6, 0.3),
                sun_intensity=0.5,
                ambient_color=(0.4, 0.3, 0.3),
                ambient_intensity=0.3,
                sky_color_zenith=(0.3, 0.3, 0.6),
                sky_color_horizon=(1.0, 0.5, 0.2),
                fog_color=(0.8, 0.5, 0.3),
                fog_density=0.05,
                shadow_strength=0.7,
            )),
            # Dusk (20:00)
            TODKeyframe(20.0, TODLighting(
                sun_color=(0.8, 0.4, 0.2),
                sun_intensity=0.1,
                ambient_color=(0.15, 0.1, 0.2),
                ambient_intensity=0.15,
                sky_color_zenith=(0.1, 0.1, 0.25),
                sky_color_horizon=(0.4, 0.2, 0.3),
                fog_color=(0.2, 0.15, 0.2),
                fog_density=0.08,
                moon_intensity=0.1,
                shadow_strength=0.4,
            )),
            # Night begins (22:00)
            TODKeyframe(22.0, TODLighting(
                sun_color=(0.0, 0.0, 0.0),
                sun_intensity=0.0,
                ambient_color=(0.05, 0.05, 0.1),
                ambient_intensity=0.1,
                sky_color_zenith=(0.02, 0.02, 0.05),
                sky_color_horizon=(0.05, 0.05, 0.1),
                fog_color=(0.02, 0.02, 0.05),
                fog_density=0.05,
                moon_color=(0.6, 0.7, 0.8),
                moon_intensity=0.15,
                shadow_strength=0.3,
            )),
        ])

    @staticmethod
    def stylized() -> TODCurve:
        """Create a more stylized, vibrant day/night cycle."""
        return TODCurve([
            TODKeyframe(0.0, TODLighting(
                sun_intensity=0.0,
                ambient_color=(0.1, 0.1, 0.2),
                ambient_intensity=0.2,
                sky_color_zenith=(0.05, 0.05, 0.15),
                sky_color_horizon=(0.1, 0.1, 0.2),
                moon_intensity=0.3,
            )),
            TODKeyframe(6.0, TODLighting(
                sun_color=(1.0, 0.7, 0.5),
                sun_intensity=0.6,
                ambient_color=(0.5, 0.4, 0.5),
                ambient_intensity=0.35,
                sky_color_zenith=(0.5, 0.5, 0.9),
                sky_color_horizon=(1.0, 0.6, 0.4),
            )),
            TODKeyframe(12.0, TODLighting(
                sun_color=(1.0, 1.0, 0.95),
                sun_intensity=1.2,
                ambient_color=(0.6, 0.65, 0.8),
                ambient_intensity=0.5,
                sky_color_zenith=(0.4, 0.6, 1.0),
                sky_color_horizon=(0.7, 0.85, 1.0),
            )),
            TODKeyframe(18.0, TODLighting(
                sun_color=(1.0, 0.5, 0.2),
                sun_intensity=0.6,
                ambient_color=(0.5, 0.35, 0.4),
                ambient_intensity=0.35,
                sky_color_zenith=(0.4, 0.3, 0.7),
                sky_color_horizon=(1.0, 0.4, 0.2),
            )),
        ])

    @staticmethod
    def always_noon() -> TODCurve:
        """Create a static noon lighting (no day/night cycle)."""
        noon_lighting = TODLighting(
            sun_color=(1.0, 0.98, 0.95),
            sun_intensity=1.0,
            ambient_color=(0.55, 0.6, 0.7),
            ambient_intensity=0.4,
            sky_color_zenith=(0.3, 0.5, 0.95),
            sky_color_horizon=(0.6, 0.75, 0.95),
        )
        return TODCurve([TODKeyframe(12.0, noon_lighting)])

    @staticmethod
    def always_night() -> TODCurve:
        """Create a static night lighting."""
        night_lighting = TODLighting(
            sun_intensity=0.0,
            ambient_color=(0.05, 0.05, 0.1),
            ambient_intensity=0.1,
            sky_color_zenith=(0.02, 0.02, 0.05),
            sky_color_horizon=(0.05, 0.05, 0.1),
            moon_intensity=0.2,
        )
        return TODCurve([TODKeyframe(0.0, night_lighting)])


class TimeOfDayController:
    """
    Controller for time-of-day simulation.

    Manages time progression, sun position calculation,
    and lighting interpolation.
    """

    def __init__(
        self,
        time_hours: float = 12.0,
        time_scale: float = 1.0,
        latitude: float = 45.0,
        day_of_year: int = 172,  # June 21 (summer solstice in northern hemisphere)
        lighting_curve: Optional[TODCurve] = None,
    ) -> None:
        """
        Initialize time-of-day controller.

        Args:
            time_hours: Starting time (0-24).
            time_scale: Time multiplier (1 = realtime, 60 = 1 hour per minute).
            latitude: Geographic latitude for sun calculations (-90 to 90).
            day_of_year: Day of year (1-365) for seasonal variation.
            lighting_curve: Lighting curve for interpolation.
        """
        self._time_hours = time_hours % 24.0
        self._day_count = 0
        self._time_scale = time_scale
        self._latitude = max(-90.0, min(90.0, latitude))
        self._day_of_year = max(1, min(365, day_of_year))
        self._paused = False
        self._lighting_curve = lighting_curve or TimeOfDayPreset.realistic()

        # Callbacks
        self._period_callbacks: List[TimeCallback] = []
        self._previous_period: Optional[TimeOfDayPeriod] = None

    @property
    def time_hours(self) -> float:
        """Get current time in hours (0-24)."""
        return self._time_hours

    @property
    def day_count(self) -> int:
        """Get number of days elapsed."""
        return self._day_count

    @property
    def time_scale(self) -> float:
        """Get time scale multiplier."""
        return self._time_scale

    @time_scale.setter
    def time_scale(self, value: float) -> None:
        """Set time scale multiplier."""
        self._time_scale = max(0.0, value)

    @property
    def latitude(self) -> float:
        """Get geographic latitude."""
        return self._latitude

    @latitude.setter
    def latitude(self, value: float) -> None:
        """Set geographic latitude."""
        self._latitude = max(-90.0, min(90.0, value))

    @property
    def day_of_year(self) -> int:
        """Get day of year."""
        return self._day_of_year

    @day_of_year.setter
    def day_of_year(self, value: int) -> None:
        """Set day of year."""
        self._day_of_year = max(1, min(365, value))

    @property
    def paused(self) -> bool:
        """Check if time is paused."""
        return self._paused

    @paused.setter
    def paused(self, value: bool) -> None:
        """Set paused state."""
        self._paused = value

    def update(self, dt: float) -> None:
        """
        Update time progression.

        Args:
            dt: Delta time in seconds.
        """
        if self._paused:
            return

        # Convert dt to hours and apply scale
        hours_delta = (dt / 3600.0) * self._time_scale
        self._time_hours += hours_delta

        # Handle day wrap
        while self._time_hours >= 24.0:
            self._time_hours -= 24.0
            self._day_count += 1
            self._day_of_year = ((self._day_of_year - 1 + 1) % 365) + 1

        # Check for period change
        current_period = self.get_period()
        if self._previous_period is not None and current_period != self._previous_period:
            for callback in self._period_callbacks:
                try:
                    callback(self._previous_period, current_period)
                except Exception:
                    pass
        self._previous_period = current_period

    def set_time(self, hours: float) -> None:
        """
        Set current time.

        Args:
            hours: Time in hours (0-24).
        """
        self._time_hours = hours % 24.0
        self._previous_period = self.get_period()

    def get_time(self) -> float:
        """Get current time in hours."""
        return self._time_hours

    def get_time_string(self) -> str:
        """Get current time as formatted string (HH:MM)."""
        hours = int(self._time_hours)
        minutes = int((self._time_hours - hours) * 60)
        return f"{hours:02d}:{minutes:02d}"

    def get_period(self) -> TimeOfDayPeriod:
        """
        Get current time-of-day period.

        Returns:
            Current period (DAWN, MORNING, NOON, etc.).
        """
        time = self._time_hours

        # Handle night first (21:00 - 4:30, wraps around midnight)
        # This must be checked before the loop since night spans midnight
        if time >= 21.0 or time < 4.5:
            return TimeOfDayPeriod.NIGHT

        # Check each period (all periods except NIGHT have simple ranges)
        for period, (start, end) in PERIOD_BOUNDARIES.items():
            if period != TimeOfDayPeriod.NIGHT and start <= time < end:
                return period

        # Fallback - should not be reached for valid times 0-24
        return TimeOfDayPeriod.NIGHT

    def get_sun_position(self) -> SunPosition:
        """
        Calculate sun position based on time, latitude, and day of year.

        Uses astronomical formulas for realistic sun path.

        Returns:
            Sun position (azimuth and elevation).
        """
        # Convert time to solar hour angle
        solar_time = self._time_hours
        hour_angle = (solar_time - 12.0) * 15.0  # degrees

        # Calculate solar declination (angle of sun relative to equator)
        # Simplified formula: peaks at summer solstice, lowest at winter solstice
        day_angle = 2.0 * math.pi * (self._day_of_year - 1) / 365.0
        declination = math.radians(
            23.45 * math.sin(day_angle + math.pi / 2 - 2 * math.pi * 81 / 365)
        )

        # Convert latitude to radians
        lat_rad = math.radians(self._latitude)

        # Calculate solar elevation angle
        sin_elevation = (
            math.sin(lat_rad) * math.sin(declination) +
            math.cos(lat_rad) * math.cos(declination) * math.cos(math.radians(hour_angle))
        )
        elevation = math.degrees(math.asin(max(-1, min(1, sin_elevation))))

        # Calculate solar azimuth angle
        cos_azimuth = (
            (math.sin(declination) - math.sin(lat_rad) * sin_elevation) /
            (math.cos(lat_rad) * math.cos(math.radians(elevation)) + 1e-10)
        )
        cos_azimuth = max(-1, min(1, cos_azimuth))
        azimuth = math.degrees(math.acos(cos_azimuth))

        # Adjust azimuth based on time of day
        if solar_time > 12.0:
            azimuth = 360.0 - azimuth

        return SunPosition(azimuth=azimuth, elevation=elevation)

    def get_lighting(self) -> TODLighting:
        """
        Get current lighting configuration.

        Returns:
            Interpolated lighting for current time.
        """
        return self._lighting_curve.interpolate(self._time_hours)

    def set_lighting_curve(self, curve: TODCurve) -> None:
        """Set the lighting curve."""
        self._lighting_curve = curve

    def add_period_callback(self, callback: TimeCallback) -> None:
        """Add a callback for period changes."""
        self._period_callbacks.append(callback)

    def remove_period_callback(self, callback: TimeCallback) -> None:
        """Remove a period change callback."""
        try:
            self._period_callbacks.remove(callback)
        except ValueError:
            pass

    def get_normalized_time(self) -> float:
        """Get time as normalized value (0-1)."""
        return self._time_hours / 24.0

    def is_daytime(self) -> bool:
        """Check if it's currently daytime."""
        sun_pos = self.get_sun_position()
        return sun_pos.is_day

    def get_sun_direction(self) -> Tuple[float, float, float]:
        """Get sun direction vector."""
        return self.get_sun_position().direction


__all__ = [
    # Enums
    "TimeOfDayPeriod",
    # Data classes
    "SunPosition",
    "TODLighting",
    "TODKeyframe",
    # Classes
    "TODCurve",
    "TimeOfDayPreset",
    "TimeOfDayController",
    # Constants
    "PERIOD_BOUNDARIES",
    # Types
    "Color3",
    "TimeCallback",
]
