"""
Environment Lighting System - Dynamic environmental lighting.

Provides directional lights for sun and moon, ambient lighting,
and integration with time-of-day and weather systems.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from engine.world.environment.time_of_day import (
    TimeOfDayController,
    TODLighting,
    SunPosition,
)
from engine.world.environment.weather import WeatherParameters
from engine.world.environment.sky import SkyManager

# Type aliases
Color3 = Tuple[float, float, float]
Direction3 = Tuple[float, float, float]
Observer = Callable[[str, Any, Any], None]


@dataclass
class DirectionalLight:
    """
    A directional light source (sun, moon, etc.).

    Directional lights have no position, only direction,
    and illuminate the entire scene uniformly.
    """
    direction: Direction3 = (0.0, -1.0, 0.0)  # Points toward light source
    color: Color3 = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    casts_shadows: bool = True

    # Shadow settings
    shadow_cascade_count: int = 4
    shadow_distance: float = 500.0
    shadow_bias: float = 0.001
    shadow_softness: float = 0.5

    # Observers
    _observers: List[Observer] = field(default_factory=list, repr=False)

    def set_direction_from_angles(self, azimuth: float, elevation: float) -> None:
        """
        Set direction from azimuth and elevation angles.

        Args:
            azimuth: Horizontal angle in degrees (0 = North).
            elevation: Vertical angle in degrees (0 = horizon).
        """
        az_rad = math.radians(azimuth)
        el_rad = math.radians(elevation)

        # Direction pointing toward the light (opposite of shadow direction)
        old_direction = self.direction
        self.direction = (
            math.sin(az_rad) * math.cos(el_rad),
            math.sin(el_rad),
            math.cos(az_rad) * math.cos(el_rad),
        )
        self._notify("direction", old_direction, self.direction)

    def get_shadow_direction(self) -> Direction3:
        """Get direction shadows should be cast (opposite of light direction)."""
        return (
            -self.direction[0],
            -self.direction[1],
            -self.direction[2],
        )

    def add_observer(self, observer: Observer) -> None:
        """Add an observer for property changes."""
        self._observers.append(observer)

    def _notify(self, prop: str, old_val: Any, new_val: Any) -> None:
        """Notify observers of a property change."""
        for obs in self._observers:
            try:
                obs(prop, old_val, new_val)
            except Exception:
                pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "direction": self.direction,
            "color": self.color,
            "intensity": self.intensity,
            "casts_shadows": self.casts_shadows,
            "shadow_cascade_count": self.shadow_cascade_count,
            "shadow_distance": self.shadow_distance,
            "shadow_bias": self.shadow_bias,
            "shadow_softness": self.shadow_softness,
        }

    def lerp(self, other: "DirectionalLight", t: float) -> "DirectionalLight":
        """
        Interpolate between two lights.

        Args:
            other: Target light.
            t: Interpolation factor (0-1).

        Returns:
            Interpolated light.
        """
        t = max(0.0, min(1.0, t))
        inv_t = 1.0 - t

        # Interpolate direction (normalize after)
        new_dir = (
            self.direction[0] * inv_t + other.direction[0] * t,
            self.direction[1] * inv_t + other.direction[1] * t,
            self.direction[2] * inv_t + other.direction[2] * t,
        )
        length = math.sqrt(new_dir[0]**2 + new_dir[1]**2 + new_dir[2]**2)
        if length > 0:
            new_dir = (new_dir[0]/length, new_dir[1]/length, new_dir[2]/length)

        return DirectionalLight(
            direction=new_dir,
            color=(
                self.color[0] * inv_t + other.color[0] * t,
                self.color[1] * inv_t + other.color[1] * t,
                self.color[2] * inv_t + other.color[2] * t,
            ),
            intensity=self.intensity * inv_t + other.intensity * t,
            casts_shadows=self.casts_shadows or other.casts_shadows,
            shadow_cascade_count=max(self.shadow_cascade_count, other.shadow_cascade_count),
            shadow_distance=self.shadow_distance * inv_t + other.shadow_distance * t,
            shadow_bias=self.shadow_bias * inv_t + other.shadow_bias * t,
            shadow_softness=self.shadow_softness * inv_t + other.shadow_softness * t,
        )


@dataclass
class AmbientLight:
    """
    Ambient lighting for the scene.

    Provides uniform illumination to all surfaces,
    optionally with separate sky/ground colors for
    hemisphere lighting.
    """
    color: Color3 = (0.5, 0.6, 0.7)
    intensity: float = 0.3

    # Hemisphere lighting (optional)
    sky_color: Optional[Color3] = None
    ground_color: Optional[Color3] = None

    # Ambient occlusion
    ao_enabled: bool = True
    ao_intensity: float = 0.5
    ao_radius: float = 0.5

    # Observers
    _observers: List[Observer] = field(default_factory=list, repr=False)

    @property
    def is_hemisphere(self) -> bool:
        """Check if using hemisphere lighting."""
        return self.sky_color is not None and self.ground_color is not None

    def get_light_at_normal(self, normal: Direction3) -> Color3:
        """
        Get ambient light contribution for a surface normal.

        Args:
            normal: Surface normal (normalized).

        Returns:
            RGB color contribution.
        """
        if not self.is_hemisphere:
            return (
                self.color[0] * self.intensity,
                self.color[1] * self.intensity,
                self.color[2] * self.intensity,
            )

        # Blend between sky and ground based on normal.y
        sky_factor = (normal[1] + 1.0) * 0.5  # 0 at bottom, 1 at top
        ground_factor = 1.0 - sky_factor

        r = (self.sky_color[0] * sky_factor + self.ground_color[0] * ground_factor) * self.intensity
        g = (self.sky_color[1] * sky_factor + self.ground_color[1] * ground_factor) * self.intensity
        b = (self.sky_color[2] * sky_factor + self.ground_color[2] * ground_factor) * self.intensity

        return (r, g, b)

    def add_observer(self, observer: Observer) -> None:
        """Add an observer for property changes."""
        self._observers.append(observer)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "color": self.color,
            "intensity": self.intensity,
            "sky_color": self.sky_color,
            "ground_color": self.ground_color,
            "ao_enabled": self.ao_enabled,
            "ao_intensity": self.ao_intensity,
            "ao_radius": self.ao_radius,
        }

    def lerp(self, other: "AmbientLight", t: float) -> "AmbientLight":
        """Interpolate between two ambient lights."""
        t = max(0.0, min(1.0, t))
        inv_t = 1.0 - t

        def lerp_color(a: Optional[Color3], b: Optional[Color3]) -> Optional[Color3]:
            if a is None or b is None:
                return b if t > 0.5 else a
            return (
                a[0] * inv_t + b[0] * t,
                a[1] * inv_t + b[1] * t,
                a[2] * inv_t + b[2] * t,
            )

        return AmbientLight(
            color=(
                self.color[0] * inv_t + other.color[0] * t,
                self.color[1] * inv_t + other.color[1] * t,
                self.color[2] * inv_t + other.color[2] * t,
            ),
            intensity=self.intensity * inv_t + other.intensity * t,
            sky_color=lerp_color(self.sky_color, other.sky_color),
            ground_color=lerp_color(self.ground_color, other.ground_color),
            ao_enabled=self.ao_enabled or other.ao_enabled,
            ao_intensity=self.ao_intensity * inv_t + other.ao_intensity * t,
            ao_radius=self.ao_radius * inv_t + other.ao_radius * t,
        )


class SunLight(DirectionalLight):
    """
    Sun light source linked to time of day.

    Automatically updates direction and color based on
    time-of-day controller.
    """

    def __init__(
        self,
        linked_to_tod: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize sun light.

        Args:
            linked_to_tod: If True, automatically update from TOD.
            **kwargs: DirectionalLight parameters.
        """
        super().__init__(**kwargs)
        self.linked_to_tod = linked_to_tod
        self._base_color: Color3 = self.color
        self._base_intensity: float = self.intensity

    def update_from_tod(self, tod_controller: TimeOfDayController) -> None:
        """
        Update sun light from time-of-day controller.

        Args:
            tod_controller: Time of day controller.
        """
        if not self.linked_to_tod:
            return

        # Get sun position
        sun_pos = tod_controller.get_sun_position()
        self.set_direction_from_angles(sun_pos.azimuth, sun_pos.elevation)

        # Get lighting parameters
        lighting = tod_controller.get_lighting()
        old_color = self.color
        old_intensity = self.intensity

        self.color = lighting.sun_color
        self.intensity = lighting.sun_intensity

        # Reduce intensity when sun is below horizon
        if sun_pos.elevation < 0:
            self.intensity = 0.0
        elif sun_pos.elevation < 10:
            self.intensity *= sun_pos.elevation / 10.0

        if old_color != self.color:
            self._notify("color", old_color, self.color)
        if old_intensity != self.intensity:
            self._notify("intensity", old_intensity, self.intensity)


class MoonLight(DirectionalLight):
    """
    Moon light source with phase support.

    Provides subtle illumination at night based on
    moon phase and position.
    """

    def __init__(
        self,
        moon_phase: float = 0.5,  # 0 = new, 0.5 = full
        **kwargs: Any,
    ) -> None:
        """
        Initialize moon light.

        Args:
            moon_phase: Moon phase (0-1).
            **kwargs: DirectionalLight parameters.
        """
        # Default moon settings
        kwargs.setdefault("color", (0.6, 0.7, 0.8))
        kwargs.setdefault("intensity", 0.2)  # Base intensity before phase adjustment
        kwargs.setdefault("shadow_softness", 0.8)
        super().__init__(**kwargs)

        self._moon_phase = moon_phase % 1.0
        self._base_intensity = self.intensity
        self._update_intensity_from_phase()  # Apply phase on construction

    @property
    def moon_phase(self) -> float:
        """Get moon phase (0-1)."""
        return self._moon_phase

    @moon_phase.setter
    def moon_phase(self, value: float) -> None:
        """Set moon phase (0-1)."""
        old_phase = self._moon_phase
        self._moon_phase = value % 1.0
        self._update_intensity_from_phase()
        self._notify("moon_phase", old_phase, self._moon_phase)

    def _update_intensity_from_phase(self) -> None:
        """Update intensity based on moon phase."""
        # Full moon (0.5) = max brightness
        # New moon (0 or 1) = min brightness
        phase_factor = 1.0 - abs(self._moon_phase - 0.5) * 2.0
        self.intensity = self._base_intensity * phase_factor

    def update_from_tod(self, tod_controller: TimeOfDayController) -> None:
        """
        Update moon light from time-of-day controller.

        Args:
            tod_controller: Time of day controller.
        """
        # Moon is roughly opposite to sun
        sun_pos = tod_controller.get_sun_position()
        moon_azimuth = (sun_pos.azimuth + 180.0) % 360.0
        moon_elevation = -sun_pos.elevation * 0.8  # Not exactly opposite

        # Only visible when elevation > 0
        if moon_elevation > 0:
            self.set_direction_from_angles(moon_azimuth, moon_elevation)
            self._update_intensity_from_phase()

            # Further reduce intensity during day
            if sun_pos.elevation > 0:
                self.intensity *= max(0.0, 1.0 - sun_pos.elevation / 30.0)
        else:
            self.intensity = 0.0

        # Get lighting parameters for moon color
        lighting = tod_controller.get_lighting()
        old_color = self.color
        self.color = lighting.moon_color
        if old_color != self.color:
            self._notify("color", old_color, self.color)


class EnvironmentLighting:
    """
    Complete environment lighting system.

    Manages sun, moon, and ambient lighting with
    time-of-day and weather integration.
    """

    def __init__(
        self,
        sky_manager: Optional[SkyManager] = None,
    ) -> None:
        """
        Initialize environment lighting.

        Args:
            sky_manager: Sky manager for atmosphere integration.
        """
        self.sun = SunLight(
            linked_to_tod=True,
            color=(1.0, 0.95, 0.9),
            intensity=1.0,
            casts_shadows=True,
        )

        self.moon = MoonLight(
            moon_phase=0.5,
            casts_shadows=True,
            shadow_cascade_count=2,
        )

        self.ambient = AmbientLight(
            color=(0.5, 0.6, 0.7),
            intensity=0.3,
        )

        self.sky_manager = sky_manager

        # Weather modifiers
        self._weather_cloud_dimming: float = 1.0
        self._weather_fog_intensity: float = 0.0

        # Cached main light
        self._main_light: DirectionalLight = self.sun

    def update(
        self,
        tod_controller: Optional[TimeOfDayController] = None,
        weather_params: Optional[WeatherParameters] = None,
    ) -> None:
        """
        Update environment lighting.

        Args:
            tod_controller: Time of day controller.
            weather_params: Current weather parameters.
        """
        if tod_controller:
            # Update sun and moon from TOD
            self.sun.update_from_tod(tod_controller)
            self.moon.update_from_tod(tod_controller)

            # Update ambient from TOD lighting
            lighting = tod_controller.get_lighting()
            self.ambient.color = lighting.ambient_color
            self.ambient.intensity = lighting.ambient_intensity
            self.ambient.sky_color = lighting.sky_color_zenith
            self.ambient.ground_color = (0.2, 0.15, 0.1)  # Dark ground color

            # Update sky manager
            if self.sky_manager:
                self.sky_manager.update(tod_controller)

        if weather_params:
            self.apply_weather_modifiers(weather_params)

        # Update main light
        self._main_light = self._determine_main_light()

    def apply_weather_modifiers(self, weather_params: WeatherParameters) -> None:
        """
        Apply weather effects to lighting.

        Args:
            weather_params: Current weather parameters.
        """
        # Cloud cover dims sunlight
        self._weather_cloud_dimming = 1.0 - weather_params.cloud_density * 0.7

        # Fog affects visibility
        self._weather_fog_intensity = weather_params.fog_density

        # Apply to sun
        old_sun_intensity = self.sun.intensity
        self.sun.intensity *= self._weather_cloud_dimming

        # Heavy precipitation further dims light
        if weather_params.precipitation > 0.5:
            self.sun.intensity *= 1.0 - (weather_params.precipitation - 0.5)

        # Increase ambient during overcast
        if weather_params.cloud_density > 0.7:
            # Overcast conditions have more diffuse light
            self.ambient.intensity = min(0.5, self.ambient.intensity * 1.3)

        # Fog tints ambient color
        if weather_params.fog_density > 0.3:
            fog_factor = weather_params.fog_density * 0.5
            self.ambient.color = (
                self.ambient.color[0] * (1 - fog_factor) + 0.6 * fog_factor,
                self.ambient.color[1] * (1 - fog_factor) + 0.65 * fog_factor,
                self.ambient.color[2] * (1 - fog_factor) + 0.7 * fog_factor,
            )

    def get_main_light(self) -> DirectionalLight:
        """
        Get the primary light source (sun during day, moon at night).

        Returns:
            The main directional light.
        """
        return self._main_light

    def _determine_main_light(self) -> DirectionalLight:
        """Determine which light is the main light source."""
        if self.sun.intensity > self.moon.intensity:
            return self.sun
        elif self.moon.intensity > 0:
            return self.moon
        else:
            return self.sun

    def get_total_ambient(self, normal: Direction3 = (0, 1, 0)) -> Color3:
        """
        Get total ambient light contribution.

        Args:
            normal: Surface normal for hemisphere lighting.

        Returns:
            RGB ambient color.
        """
        return self.ambient.get_light_at_normal(normal)

    def get_fog_settings(self) -> Tuple[Color3, float]:
        """
        Get fog settings from lighting.

        Returns:
            Tuple of (fog_color, fog_density).
        """
        # Base fog color from ambient
        fog_color = (
            self.ambient.color[0] * 0.8 + 0.2,
            self.ambient.color[1] * 0.8 + 0.2,
            self.ambient.color[2] * 0.8 + 0.2,
        )
        return (fog_color, self._weather_fog_intensity)

    def to_dict(self) -> Dict[str, Any]:
        """Convert lighting state to dictionary."""
        return {
            "sun": self.sun.to_dict(),
            "moon": self.moon.to_dict(),
            "ambient": self.ambient.to_dict(),
            "weather_cloud_dimming": self._weather_cloud_dimming,
            "weather_fog_intensity": self._weather_fog_intensity,
        }

    def set_sun_enabled(self, enabled: bool) -> None:
        """Enable or disable sun light."""
        if not enabled:
            self.sun.intensity = 0.0
        else:
            self.sun.linked_to_tod = True

    def set_moon_enabled(self, enabled: bool) -> None:
        """Enable or disable moon light."""
        if not enabled:
            self.moon.intensity = 0.0

    def get_shadow_settings(self) -> Dict[str, Any]:
        """Get shadow settings from main light."""
        light = self.get_main_light()
        return {
            "enabled": light.casts_shadows and light.intensity > 0.1,
            "direction": light.get_shadow_direction(),
            "cascade_count": light.shadow_cascade_count,
            "distance": light.shadow_distance,
            "bias": light.shadow_bias,
            "softness": light.shadow_softness,
        }


class LightProbe:
    """
    Light probe for capturing environment lighting at a point.

    Used for indirect lighting and reflections.
    """

    def __init__(
        self,
        position: Tuple[float, float, float] = (0, 0, 0),
        radius: float = 10.0,
    ) -> None:
        """
        Initialize light probe.

        Args:
            position: World position.
            radius: Influence radius.
        """
        self.position = position
        self.radius = radius
        self.intensity = 1.0

        # Spherical harmonics coefficients (L2, 9 coefficients per channel)
        self._sh_coefficients: List[Color3] = [(0, 0, 0) for _ in range(9)]

        # Captured state
        self._is_captured = False
        self._capture_time: float = 0.0

    def capture(self, env_lighting: EnvironmentLighting) -> None:
        """
        Capture lighting at probe position.

        Args:
            env_lighting: Environment lighting to capture.
        """
        # Simplified capture - in real implementation this would
        # render a cubemap and compute SH coefficients

        # Use ambient as base
        ambient = env_lighting.ambient.color
        sun = env_lighting.sun

        # DC coefficient (constant term)
        self._sh_coefficients[0] = (
            ambient[0] * 0.5,
            ambient[1] * 0.5,
            ambient[2] * 0.5,
        )

        # First-order coefficients (directional)
        if sun.intensity > 0:
            dir_contrib = sun.intensity * 0.3
            self._sh_coefficients[1] = (
                sun.color[0] * sun.direction[1] * dir_contrib,
                sun.color[1] * sun.direction[1] * dir_contrib,
                sun.color[2] * sun.direction[1] * dir_contrib,
            )
            self._sh_coefficients[2] = (
                sun.color[0] * sun.direction[2] * dir_contrib,
                sun.color[1] * sun.direction[2] * dir_contrib,
                sun.color[2] * sun.direction[2] * dir_contrib,
            )
            self._sh_coefficients[3] = (
                sun.color[0] * sun.direction[0] * dir_contrib,
                sun.color[1] * sun.direction[0] * dir_contrib,
                sun.color[2] * sun.direction[0] * dir_contrib,
            )

        self._is_captured = True

    def get_irradiance(self, normal: Direction3) -> Color3:
        """
        Get irradiance for a surface normal.

        Args:
            normal: Surface normal (normalized).

        Returns:
            RGB irradiance.
        """
        if not self._is_captured:
            return (0.3, 0.3, 0.3)  # Default gray

        # Evaluate SH (simplified - just DC + first order)
        result = list(self._sh_coefficients[0])

        # First order
        result[0] += self._sh_coefficients[1][0] * normal[1]
        result[1] += self._sh_coefficients[1][1] * normal[1]
        result[2] += self._sh_coefficients[1][2] * normal[1]

        result[0] += self._sh_coefficients[2][0] * normal[2]
        result[1] += self._sh_coefficients[2][1] * normal[2]
        result[2] += self._sh_coefficients[2][2] * normal[2]

        result[0] += self._sh_coefficients[3][0] * normal[0]
        result[1] += self._sh_coefficients[3][1] * normal[0]
        result[2] += self._sh_coefficients[3][2] * normal[0]

        return (
            max(0, result[0]) * self.intensity,
            max(0, result[1]) * self.intensity,
            max(0, result[2]) * self.intensity,
        )

    def get_blend_weight(self, point: Tuple[float, float, float]) -> float:
        """Get blend weight for a point."""
        dx = point[0] - self.position[0]
        dy = point[1] - self.position[1]
        dz = point[2] - self.position[2]
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)

        if distance >= self.radius:
            return 0.0

        # Smoothstep falloff
        t = 1.0 - distance / self.radius
        return t * t * (3 - 2 * t)


class LightProbeGrid:
    """
    Grid of light probes for spatial indirect lighting.
    """

    def __init__(
        self,
        bounds_min: Tuple[float, float, float] = (-100, 0, -100),
        bounds_max: Tuple[float, float, float] = (100, 50, 100),
        resolution: Tuple[int, int, int] = (5, 3, 5),
    ) -> None:
        """
        Initialize light probe grid.

        Args:
            bounds_min: Minimum bounds.
            bounds_max: Maximum bounds.
            resolution: Grid resolution (x, y, z).
        """
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max
        self.resolution = resolution

        # Generate probes
        self.probes: List[LightProbe] = []
        self._generate_probes()

    def _generate_probes(self) -> None:
        """Generate probe grid."""
        self.probes.clear()

        for x in range(self.resolution[0]):
            for y in range(self.resolution[1]):
                for z in range(self.resolution[2]):
                    # Calculate position
                    t_x = x / max(1, self.resolution[0] - 1)
                    t_y = y / max(1, self.resolution[1] - 1)
                    t_z = z / max(1, self.resolution[2] - 1)

                    pos = (
                        self.bounds_min[0] + t_x * (self.bounds_max[0] - self.bounds_min[0]),
                        self.bounds_min[1] + t_y * (self.bounds_max[1] - self.bounds_min[1]),
                        self.bounds_min[2] + t_z * (self.bounds_max[2] - self.bounds_min[2]),
                    )

                    # Calculate radius based on grid spacing
                    spacing = (
                        (self.bounds_max[0] - self.bounds_min[0]) / max(1, self.resolution[0] - 1),
                        (self.bounds_max[1] - self.bounds_min[1]) / max(1, self.resolution[1] - 1),
                        (self.bounds_max[2] - self.bounds_min[2]) / max(1, self.resolution[2] - 1),
                    )
                    radius = max(spacing) * 1.5

                    self.probes.append(LightProbe(position=pos, radius=radius))

    def capture_all(self, env_lighting: EnvironmentLighting) -> None:
        """Capture lighting for all probes."""
        for probe in self.probes:
            probe.capture(env_lighting)

    def get_irradiance_at(
        self,
        position: Tuple[float, float, float],
        normal: Direction3,
    ) -> Color3:
        """
        Get blended irradiance at a position.

        Args:
            position: World position.
            normal: Surface normal.

        Returns:
            Blended RGB irradiance.
        """
        total_weight = 0.0
        result = [0.0, 0.0, 0.0]

        for probe in self.probes:
            weight = probe.get_blend_weight(position)
            if weight > 0:
                irr = probe.get_irradiance(normal)
                result[0] += irr[0] * weight
                result[1] += irr[1] * weight
                result[2] += irr[2] * weight
                total_weight += weight

        if total_weight > 0:
            return (
                result[0] / total_weight,
                result[1] / total_weight,
                result[2] / total_weight,
            )

        return (0.3, 0.3, 0.3)  # Default


__all__ = [
    # Light types
    "DirectionalLight",
    "AmbientLight",
    "SunLight",
    "MoonLight",
    # Main system
    "EnvironmentLighting",
    # Light probes
    "LightProbe",
    "LightProbeGrid",
    # Types
    "Color3",
    "Direction3",
    "Observer",
]
