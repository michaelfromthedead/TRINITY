"""
Weather System - State machine-based weather simulation.

Provides a comprehensive weather system with state machine transitions,
parameter interpolation, and regional weather zones. Uses StateMeta
patterns for weather state management.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from engine.world.environment.constants import (
    DEFAULT_WEATHER_TRANSITION_DURATION,
    DEFAULT_WIND_SPEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_HUMIDITY,
    DEFAULT_VISIBILITY,
    DEFAULT_ATMOSPHERIC_PRESSURE,
)

# Type aliases
Point3D = Tuple[float, float, float]
WeatherCallback = Callable[["WeatherType", "WeatherType"], None]
TransitionCallback = Callable[["WeatherTransition"], None]


class WeatherType(Enum):
    """Types of weather conditions."""
    CLEAR = auto()
    CLOUDY = auto()
    RAIN = auto()
    STORM = auto()
    FOG = auto()
    SNOW = auto()
    HAIL = auto()
    SANDSTORM = auto()
    OVERCAST = auto()


@dataclass
class WeatherParameters:
    """
    Parameters defining a weather state.

    All values are normalized (0-1) unless otherwise noted.
    """
    # Precipitation
    precipitation: float = 0.0  # 0-1 intensity
    precipitation_type: str = "none"  # rain, snow, hail, none

    # Wind
    wind_speed: float = 5.0  # m/s
    wind_direction: float = 0.0  # degrees (0 = north)
    wind_gustiness: float = 0.0  # 0-1

    # Clouds
    cloud_density: float = 0.0  # 0-1
    cloud_type: str = "cumulus"  # cumulus, stratus, cirrus, etc.

    # Visibility
    fog_density: float = 0.0  # 0-1
    visibility: float = 10000.0  # meters

    # Temperature
    temperature: float = 20.0  # Celsius
    humidity: float = 0.5  # 0-1

    # Surface effects
    wetness: float = 0.0  # 0-1 surface wetness
    snow_accumulation: float = 0.0  # 0-1

    # Atmospheric
    atmospheric_pressure: float = 1013.25  # hPa
    lightning_frequency: float = 0.0  # 0-1

    def lerp(self, other: "WeatherParameters", t: float) -> "WeatherParameters":
        """
        Linearly interpolate between this and another set of parameters.

        Args:
            other: Target parameters.
            t: Interpolation factor (0 = self, 1 = other).

        Returns:
            Interpolated parameters.
        """
        t = max(0.0, min(1.0, t))
        inv_t = 1.0 - t

        return WeatherParameters(
            precipitation=self.precipitation * inv_t + other.precipitation * t,
            precipitation_type=other.precipitation_type if t > 0.5 else self.precipitation_type,
            wind_speed=self.wind_speed * inv_t + other.wind_speed * t,
            wind_direction=self._lerp_angle(self.wind_direction, other.wind_direction, t),
            wind_gustiness=self.wind_gustiness * inv_t + other.wind_gustiness * t,
            cloud_density=self.cloud_density * inv_t + other.cloud_density * t,
            cloud_type=other.cloud_type if t > 0.5 else self.cloud_type,
            fog_density=self.fog_density * inv_t + other.fog_density * t,
            visibility=self.visibility * inv_t + other.visibility * t,
            temperature=self.temperature * inv_t + other.temperature * t,
            humidity=self.humidity * inv_t + other.humidity * t,
            wetness=self.wetness * inv_t + other.wetness * t,
            snow_accumulation=self.snow_accumulation * inv_t + other.snow_accumulation * t,
            atmospheric_pressure=self.atmospheric_pressure * inv_t + other.atmospheric_pressure * t,
            lightning_frequency=self.lightning_frequency * inv_t + other.lightning_frequency * t,
        )

    @staticmethod
    def _lerp_angle(a: float, b: float, t: float) -> float:
        """Interpolate between two angles, taking the shortest path."""
        diff = ((b - a + 180) % 360) - 180
        return (a + diff * t) % 360

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary."""
        return {
            "precipitation": self.precipitation,
            "precipitation_type": self.precipitation_type,
            "wind_speed": self.wind_speed,
            "wind_direction": self.wind_direction,
            "wind_gustiness": self.wind_gustiness,
            "cloud_density": self.cloud_density,
            "cloud_type": self.cloud_type,
            "fog_density": self.fog_density,
            "visibility": self.visibility,
            "temperature": self.temperature,
            "humidity": self.humidity,
            "wetness": self.wetness,
            "snow_accumulation": self.snow_accumulation,
            "atmospheric_pressure": self.atmospheric_pressure,
            "lightning_frequency": self.lightning_frequency,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WeatherParameters":
        """Create parameters from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


@dataclass
class WeatherPreset:
    """
    Predefined weather configuration.

    Includes visual and audio references for the weather type.
    """
    weather_type: WeatherType
    parameters: WeatherParameters
    particle_system_id: Optional[str] = None
    ambient_sound_id: Optional[str] = None
    sky_texture_id: Optional[str] = None
    description: str = ""

    @staticmethod
    def create_default_presets() -> Dict[WeatherType, "WeatherPreset"]:
        """Create default weather presets."""
        return {
            WeatherType.CLEAR: WeatherPreset(
                weather_type=WeatherType.CLEAR,
                parameters=WeatherParameters(
                    precipitation=0.0,
                    wind_speed=3.0,
                    cloud_density=0.1,
                    fog_density=0.0,
                    visibility=10000.0,
                    temperature=22.0,
                    humidity=0.4,
                ),
                description="Clear, sunny weather",
            ),
            WeatherType.CLOUDY: WeatherPreset(
                weather_type=WeatherType.CLOUDY,
                parameters=WeatherParameters(
                    precipitation=0.0,
                    wind_speed=5.0,
                    cloud_density=0.7,
                    cloud_type="stratus",
                    fog_density=0.05,
                    visibility=8000.0,
                    temperature=18.0,
                    humidity=0.6,
                ),
                description="Cloudy with overcast skies",
            ),
            WeatherType.RAIN: WeatherPreset(
                weather_type=WeatherType.RAIN,
                parameters=WeatherParameters(
                    precipitation=0.6,
                    precipitation_type="rain",
                    wind_speed=8.0,
                    wind_gustiness=0.3,
                    cloud_density=0.9,
                    cloud_type="nimbus",
                    fog_density=0.15,
                    visibility=3000.0,
                    temperature=15.0,
                    humidity=0.85,
                    wetness=0.7,
                ),
                particle_system_id="particles/rain",
                ambient_sound_id="sounds/rain_loop",
                description="Light to moderate rain",
            ),
            WeatherType.STORM: WeatherPreset(
                weather_type=WeatherType.STORM,
                parameters=WeatherParameters(
                    precipitation=0.9,
                    precipitation_type="rain",
                    wind_speed=20.0,
                    wind_gustiness=0.7,
                    cloud_density=1.0,
                    cloud_type="cumulonimbus",
                    fog_density=0.3,
                    visibility=1000.0,
                    temperature=12.0,
                    humidity=0.95,
                    wetness=1.0,
                    lightning_frequency=0.5,
                ),
                particle_system_id="particles/heavy_rain",
                ambient_sound_id="sounds/storm_loop",
                description="Heavy thunderstorm",
            ),
            WeatherType.FOG: WeatherPreset(
                weather_type=WeatherType.FOG,
                parameters=WeatherParameters(
                    precipitation=0.0,
                    wind_speed=1.0,
                    cloud_density=0.3,
                    fog_density=0.8,
                    visibility=200.0,
                    temperature=10.0,
                    humidity=0.95,
                    wetness=0.2,
                ),
                ambient_sound_id="sounds/fog_ambience",
                description="Dense fog with low visibility",
            ),
            WeatherType.SNOW: WeatherPreset(
                weather_type=WeatherType.SNOW,
                parameters=WeatherParameters(
                    precipitation=0.5,
                    precipitation_type="snow",
                    wind_speed=6.0,
                    wind_gustiness=0.2,
                    cloud_density=0.85,
                    fog_density=0.2,
                    visibility=2000.0,
                    temperature=-5.0,
                    humidity=0.7,
                    snow_accumulation=0.3,
                ),
                particle_system_id="particles/snow",
                ambient_sound_id="sounds/wind_snow",
                description="Light to moderate snowfall",
            ),
            WeatherType.HAIL: WeatherPreset(
                weather_type=WeatherType.HAIL,
                parameters=WeatherParameters(
                    precipitation=0.7,
                    precipitation_type="hail",
                    wind_speed=15.0,
                    wind_gustiness=0.5,
                    cloud_density=0.95,
                    cloud_type="cumulonimbus",
                    fog_density=0.1,
                    visibility=2500.0,
                    temperature=5.0,
                    humidity=0.8,
                ),
                particle_system_id="particles/hail",
                ambient_sound_id="sounds/hail_loop",
                description="Hailstorm",
            ),
            WeatherType.SANDSTORM: WeatherPreset(
                weather_type=WeatherType.SANDSTORM,
                parameters=WeatherParameters(
                    precipitation=0.0,
                    wind_speed=25.0,
                    wind_gustiness=0.8,
                    cloud_density=0.2,
                    fog_density=0.7,
                    visibility=100.0,
                    temperature=35.0,
                    humidity=0.1,
                ),
                particle_system_id="particles/sandstorm",
                ambient_sound_id="sounds/sandstorm_loop",
                description="Intense sandstorm",
            ),
            WeatherType.OVERCAST: WeatherPreset(
                weather_type=WeatherType.OVERCAST,
                parameters=WeatherParameters(
                    precipitation=0.0,
                    wind_speed=4.0,
                    cloud_density=1.0,
                    cloud_type="stratus",
                    fog_density=0.0,
                    visibility=7000.0,
                    temperature=16.0,
                    humidity=0.65,
                ),
                description="Completely overcast skies",
            ),
        }


@dataclass
class WeatherTransition:
    """
    Represents an active weather transition.

    Manages interpolation between two weather states over time.
    """
    from_type: WeatherType
    to_type: WeatherType
    from_params: WeatherParameters
    to_params: WeatherParameters
    duration: float  # seconds
    elapsed: float = 0.0
    easing: str = "smoothstep"  # linear, smoothstep, ease_in, ease_out

    @property
    def progress(self) -> float:
        """Get normalized progress (0-1)."""
        if self.duration <= 0:
            return 1.0
        return min(1.0, self.elapsed / self.duration)

    @property
    def is_complete(self) -> bool:
        """Check if transition is complete."""
        return self.elapsed >= self.duration

    def get_eased_progress(self) -> float:
        """Get progress with easing applied."""
        t = self.progress

        if self.easing == "smoothstep":
            return t * t * (3 - 2 * t)
        elif self.easing == "ease_in":
            return t * t
        elif self.easing == "ease_out":
            return 1 - (1 - t) * (1 - t)
        else:  # linear
            return t

    def blend_parameters(self) -> WeatherParameters:
        """Get current blended parameters."""
        return self.from_params.lerp(self.to_params, self.get_eased_progress())

    def update(self, dt: float) -> None:
        """Update transition progress."""
        self.elapsed += dt


class WeatherStateMachine:
    """
    State machine for weather management.

    Handles weather state transitions with validation,
    interpolation, and callback notifications.
    """

    # Valid state transitions (from -> allowed targets)
    DEFAULT_TRANSITIONS: Dict[WeatherType, Set[WeatherType]] = {
        WeatherType.CLEAR: {WeatherType.CLOUDY, WeatherType.FOG, WeatherType.OVERCAST},
        WeatherType.CLOUDY: {WeatherType.CLEAR, WeatherType.RAIN, WeatherType.OVERCAST, WeatherType.FOG},
        WeatherType.RAIN: {WeatherType.CLOUDY, WeatherType.STORM, WeatherType.FOG},
        WeatherType.STORM: {WeatherType.RAIN, WeatherType.HAIL},
        WeatherType.FOG: {WeatherType.CLEAR, WeatherType.CLOUDY, WeatherType.RAIN},
        WeatherType.SNOW: {WeatherType.CLOUDY, WeatherType.FOG, WeatherType.HAIL},
        WeatherType.HAIL: {WeatherType.STORM, WeatherType.RAIN, WeatherType.SNOW},
        WeatherType.SANDSTORM: {WeatherType.CLEAR, WeatherType.CLOUDY},
        WeatherType.OVERCAST: {WeatherType.CLEAR, WeatherType.CLOUDY, WeatherType.RAIN},
    }

    def __init__(
        self,
        initial_type: WeatherType = WeatherType.CLEAR,
        presets: Optional[Dict[WeatherType, WeatherPreset]] = None,
        valid_transitions: Optional[Dict[WeatherType, Set[WeatherType]]] = None,
    ) -> None:
        """
        Initialize weather state machine.

        Args:
            initial_type: Starting weather type.
            presets: Weather presets (uses defaults if not provided).
            valid_transitions: Valid state transitions (uses defaults if not provided).
        """
        self.presets = presets or WeatherPreset.create_default_presets()
        self.valid_transitions = valid_transitions or self.DEFAULT_TRANSITIONS.copy()

        self._current_type = initial_type
        self._target_type: Optional[WeatherType] = None
        self._current_params = self.presets[initial_type].parameters
        self._target_params: Optional[WeatherParameters] = None
        self._transition: Optional[WeatherTransition] = None

        # Callbacks
        self._on_weather_changed: List[WeatherCallback] = []
        self._on_transition_start: List[TransitionCallback] = []
        self._on_transition_complete: List[TransitionCallback] = []

    @property
    def current_type(self) -> WeatherType:
        """Get current weather type."""
        return self._current_type

    @property
    def target_type(self) -> Optional[WeatherType]:
        """Get target weather type (if transitioning)."""
        return self._target_type

    @property
    def is_transitioning(self) -> bool:
        """Check if weather is currently transitioning."""
        return self._transition is not None and not self._transition.is_complete

    def can_transition_to(self, weather_type: WeatherType) -> bool:
        """
        Check if transition to weather type is valid.

        Args:
            weather_type: Target weather type.

        Returns:
            True if transition is allowed.
        """
        if weather_type == self._current_type:
            return False

        allowed = self.valid_transitions.get(self._current_type, set())
        return weather_type in allowed

    def start_transition(
        self,
        target_type: WeatherType,
        duration: float = 60.0,
        easing: str = "smoothstep",
        force: bool = False,
    ) -> bool:
        """
        Start a weather transition.

        Args:
            target_type: Target weather type.
            duration: Transition duration in seconds.
            easing: Easing function name.
            force: If True, skip transition validation.

        Returns:
            True if transition started successfully.
        """
        if not force and not self.can_transition_to(target_type):
            return False

        if target_type not in self.presets:
            return False

        self._target_type = target_type
        self._target_params = self.presets[target_type].parameters

        # If already transitioning, start from current interpolated state
        if self._transition and not self._transition.is_complete:
            from_params = self._transition.blend_parameters()
        else:
            from_params = self._current_params

        self._transition = WeatherTransition(
            from_type=self._current_type,
            to_type=target_type,
            from_params=from_params,
            to_params=self._target_params,
            duration=duration,
            easing=easing,
        )

        # Notify callbacks
        for callback in self._on_transition_start:
            try:
                callback(self._transition)
            except Exception:
                pass

        return True

    def update(self, dt: float) -> WeatherParameters:
        """
        Update weather state machine.

        Args:
            dt: Delta time in seconds.

        Returns:
            Current weather parameters.
        """
        if self._transition:
            self._transition.update(dt)

            if self._transition.is_complete:
                # Transition complete
                old_type = self._current_type
                self._current_type = self._transition.to_type
                self._current_params = self._transition.to_params
                transition = self._transition
                self._transition = None
                self._target_type = None
                self._target_params = None

                # Notify completion
                for callback in self._on_transition_complete:
                    try:
                        callback(transition)
                    except Exception:
                        pass

                # Notify weather changed
                for callback in self._on_weather_changed:
                    try:
                        callback(old_type, self._current_type)
                    except Exception:
                        pass

                return self._current_params
            else:
                return self._transition.blend_parameters()

        return self._current_params

    def get_current_parameters(self) -> WeatherParameters:
        """
        Get current weather parameters (accounting for transitions).

        Returns:
            Current interpolated weather parameters.
        """
        if self._transition and not self._transition.is_complete:
            return self._transition.blend_parameters()
        return self._current_params

    def set_weather_instant(self, weather_type: WeatherType) -> bool:
        """
        Set weather instantly without transition.

        Args:
            weather_type: Target weather type.

        Returns:
            True if successful.
        """
        if weather_type not in self.presets:
            return False

        old_type = self._current_type
        self._current_type = weather_type
        self._current_params = self.presets[weather_type].parameters
        self._transition = None
        self._target_type = None
        self._target_params = None

        # Notify
        for callback in self._on_weather_changed:
            try:
                callback(old_type, self._current_type)
            except Exception:
                pass

        return True

    def get_current_preset(self) -> WeatherPreset:
        """Get the current weather preset."""
        return self.presets[self._current_type]

    def add_weather_changed_callback(self, callback: WeatherCallback) -> None:
        """Add a callback for weather changes."""
        self._on_weather_changed.append(callback)

    def remove_weather_changed_callback(self, callback: WeatherCallback) -> None:
        """Remove a weather change callback."""
        try:
            self._on_weather_changed.remove(callback)
        except ValueError:
            pass

    def add_transition_start_callback(self, callback: TransitionCallback) -> None:
        """Add a callback for transition start."""
        self._on_transition_start.append(callback)

    def add_transition_complete_callback(self, callback: TransitionCallback) -> None:
        """Add a callback for transition completion."""
        self._on_transition_complete.append(callback)

    def get_transition_progress(self) -> float:
        """Get current transition progress (0-1), or 1.0 if not transitioning."""
        if self._transition:
            return self._transition.progress
        return 1.0

    def get_allowed_transitions(self) -> Set[WeatherType]:
        """Get all weather types that can be transitioned to from current state."""
        return self.valid_transitions.get(self._current_type, set()).copy()


@dataclass
class WeatherZone:
    """
    A zone with its own weather parameters.

    Used for regional weather that varies by location.
    """
    zone_id: str
    center: Point3D
    radius: float
    weather_type: WeatherType
    parameters: WeatherParameters
    blend_radius: float = 100.0
    priority: int = 0

    def contains_point(self, point: Point3D) -> bool:
        """Check if a point is in this zone."""
        dx = point[0] - self.center[0]
        dy = point[1] - self.center[1]
        dz = point[2] - self.center[2]
        return (dx * dx + dy * dy + dz * dz) <= (self.radius * self.radius)

    def get_blend_weight(self, point: Point3D) -> float:
        """
        Get blend weight for a point.

        Returns:
            Weight from 0.0 (at edge) to 1.0 (fully inside).
        """
        dx = point[0] - self.center[0]
        dy = point[1] - self.center[1]
        dz = point[2] - self.center[2]
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        if distance >= self.radius:
            return 0.0

        # Clamp blend_radius to not exceed radius
        effective_blend_radius = min(self.blend_radius, self.radius)
        if effective_blend_radius <= 0:
            return 1.0  # No blend zone

        inner_radius = self.radius - effective_blend_radius
        if distance <= inner_radius:
            return 1.0

        # Smoothstep blend in outer region
        t = (self.radius - distance) / effective_blend_radius
        return t * t * (3 - 2 * t)


class RegionalWeather:
    """
    Manages weather that varies by location.

    Supports multiple weather zones with blending.
    """

    def __init__(
        self,
        default_params: Optional[WeatherParameters] = None,
    ) -> None:
        """
        Initialize regional weather.

        Args:
            default_params: Default weather parameters outside all zones.
        """
        self.zones: List[WeatherZone] = []
        self._default_params = default_params or WeatherParameters()

    def add_zone(self, zone: WeatherZone) -> None:
        """Add a weather zone."""
        self.zones.append(zone)
        # Sort by priority for consistent blending
        self.zones.sort(key=lambda z: z.priority, reverse=True)

    def remove_zone(self, zone_id: str) -> Optional[WeatherZone]:
        """Remove a weather zone by ID."""
        for i, zone in enumerate(self.zones):
            if zone.zone_id == zone_id:
                return self.zones.pop(i)
        return None

    def get_zone(self, zone_id: str) -> Optional[WeatherZone]:
        """Get a zone by ID."""
        for zone in self.zones:
            if zone.zone_id == zone_id:
                return zone
        return None

    def get_weather_at(self, position: Point3D) -> WeatherParameters:
        """
        Get blended weather parameters at a position.

        Args:
            position: World position.

        Returns:
            Blended weather parameters.
        """
        if not self.zones:
            return self._default_params

        # Collect zones affecting this position
        affecting_zones: List[Tuple[WeatherZone, float]] = []
        total_weight = 0.0

        for zone in self.zones:
            weight = zone.get_blend_weight(position)
            if weight > 0:
                affecting_zones.append((zone, weight))
                total_weight += weight

        if not affecting_zones:
            return self._default_params

        # Single zone - no blending needed
        if len(affecting_zones) == 1:
            zone, weight = affecting_zones[0]
            if weight >= 1.0:
                return zone.parameters
            # Blend with default
            return self._default_params.lerp(zone.parameters, weight)

        # Multiple zones - blend based on weights
        # Start with default and blend in each zone
        result = self._default_params
        remaining_weight = 1.0

        for zone, weight in affecting_zones:
            # Normalize weight relative to remaining
            normalized_weight = min(weight, remaining_weight)
            if normalized_weight > 0:
                result = result.lerp(zone.parameters, normalized_weight / remaining_weight)
                remaining_weight -= normalized_weight
                if remaining_weight <= 0:
                    break

        return result

    def get_zones_at(self, position: Point3D) -> List[WeatherZone]:
        """Get all zones containing a position."""
        return [zone for zone in self.zones if zone.contains_point(position)]

    def clear_zones(self) -> None:
        """Remove all zones."""
        self.zones.clear()


class WeatherSystem:
    """
    Complete weather system combining state machine and regional weather.

    Provides a unified interface for weather management.
    """

    def __init__(
        self,
        initial_type: WeatherType = WeatherType.CLEAR,
        use_regional: bool = False,
    ) -> None:
        """
        Initialize weather system.

        Args:
            initial_type: Starting weather type.
            use_regional: Enable regional weather support.
        """
        self.state_machine = WeatherStateMachine(initial_type=initial_type)
        self.regional_weather: Optional[RegionalWeather] = (
            RegionalWeather() if use_regional else None
        )

    def update(self, dt: float) -> WeatherParameters:
        """Update weather system."""
        return self.state_machine.update(dt)

    def get_weather_at(self, position: Point3D) -> WeatherParameters:
        """
        Get weather at a specific position.

        Uses regional weather if enabled, otherwise returns global weather.
        """
        if self.regional_weather and self.regional_weather.zones:
            return self.regional_weather.get_weather_at(position)
        return self.state_machine.get_current_parameters()

    def transition_to(
        self,
        weather_type: WeatherType,
        duration: float = 60.0,
    ) -> bool:
        """Start a weather transition."""
        return self.state_machine.start_transition(weather_type, duration)

    def set_weather(self, weather_type: WeatherType) -> bool:
        """Set weather instantly."""
        return self.state_machine.set_weather_instant(weather_type)

    @property
    def current_type(self) -> WeatherType:
        """Get current weather type."""
        return self.state_machine.current_type


__all__ = [
    # Enums
    "WeatherType",
    # Data classes
    "WeatherParameters",
    "WeatherPreset",
    "WeatherTransition",
    "WeatherZone",
    # State machine
    "WeatherStateMachine",
    "RegionalWeather",
    "WeatherSystem",
    # Types
    "WeatherCallback",
    "TransitionCallback",
]
