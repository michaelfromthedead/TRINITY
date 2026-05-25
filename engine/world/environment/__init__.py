"""
Environment System - Volumes, weather, time of day, sky, and lighting.

This module provides comprehensive environment management for the game engine,
including:

- **Volumes**: Physics, trigger, visual, and audio volumes
- **Weather**: State machine-based weather system with transitions
- **Time of Day**: Astronomical time simulation with lighting keyframes
- **Sky**: Procedural atmosphere, HDRI, and static skybox support
- **Lighting**: Dynamic sun/moon lighting with weather integration

Usage:
    from engine.world.environment import (
        VolumeManager, TriggerVolume,
        WeatherSystem, WeatherType,
        TimeOfDayController, TimeOfDayPreset,
        SkyManager, ProceduralSky,
        EnvironmentLighting,
    )

    # Create environment systems
    tod = TimeOfDayController(time_hours=12.0)
    weather = WeatherSystem(use_regional=True)
    sky = SkyManager(sky_type="procedural")
    lighting = EnvironmentLighting(sky_manager=sky)

    # Update each frame
    def update(dt):
        tod.update(dt)
        weather.update(dt)
        sky.update(tod)
        lighting.update(tod, weather.get_weather_at(player_pos))
"""

from engine.world.environment.volumes import (
    # Enums
    VolumeType,
    VolumeShape,
    OverlapState,
    # Bounds
    BoundingBox,
    SphereBounds,
    CapsuleBounds,
    # Base
    BaseVolume,
    # Physics volumes
    PhysicsVolume,
    WaterVolume,
    PainVolume,
    KillVolume,
    # Gameplay volumes
    TriggerVolume,
    BlockingVolume,
    CameraVolume,
    SpawnVolume,
    # Visual volumes
    PostProcessVolume,
    FogVolume,
    ReflectionVolume,
    LightmassVolume,
    # Audio volumes
    ReverbVolume,
    AmbientVolume,
    # Navigation volumes
    NavModifierVolume,
    NavLinkVolume,
    NavExcludeVolume,
    # Manager
    VolumeManager,
)

from engine.world.environment.weather import (
    # Enums
    WeatherType,
    # Data classes
    WeatherParameters,
    WeatherPreset,
    WeatherTransition,
    WeatherZone,
    # Classes
    WeatherStateMachine,
    RegionalWeather,
    WeatherSystem,
)

from engine.world.environment.time_of_day import (
    # Enums
    TimeOfDayPeriod,
    # Data classes
    SunPosition,
    TODLighting,
    TODKeyframe,
    # Classes
    TODCurve,
    TimeOfDayPreset,
    TimeOfDayController,
    # Constants
    PERIOD_BOUNDARIES,
)

from engine.world.environment.sky import (
    # Data classes
    AtmosphereSettings,
    # Enums
    CelestialBodyType,
    # Sky implementations
    BaseSky,
    ProceduralSky,
    HDRISky,
    StaticSky,
    # Celestial
    CelestialBody,
    StarField,
    # Manager
    SkyManager,
)

from engine.world.environment.lighting import (
    # Light types
    DirectionalLight,
    AmbientLight,
    SunLight,
    MoonLight,
    # Main system
    EnvironmentLighting,
    # Light probes
    LightProbe,
    LightProbeGrid,
)


__all__ = [
    # === Volumes ===
    # Enums
    "VolumeType",
    "VolumeShape",
    "OverlapState",
    # Bounds
    "BoundingBox",
    "SphereBounds",
    "CapsuleBounds",
    # Base
    "BaseVolume",
    # Physics volumes
    "PhysicsVolume",
    "WaterVolume",
    "PainVolume",
    "KillVolume",
    # Gameplay volumes
    "TriggerVolume",
    "BlockingVolume",
    "CameraVolume",
    "SpawnVolume",
    # Visual volumes
    "PostProcessVolume",
    "FogVolume",
    "ReflectionVolume",
    "LightmassVolume",
    # Audio volumes
    "ReverbVolume",
    "AmbientVolume",
    # Navigation volumes
    "NavModifierVolume",
    "NavLinkVolume",
    "NavExcludeVolume",
    # Manager
    "VolumeManager",
    # === Weather ===
    "WeatherType",
    "WeatherParameters",
    "WeatherPreset",
    "WeatherTransition",
    "WeatherZone",
    "WeatherStateMachine",
    "RegionalWeather",
    "WeatherSystem",
    # === Time of Day ===
    "TimeOfDayPeriod",
    "SunPosition",
    "TODLighting",
    "TODKeyframe",
    "TODCurve",
    "TimeOfDayPreset",
    "TimeOfDayController",
    "PERIOD_BOUNDARIES",
    # === Sky ===
    "AtmosphereSettings",
    "CelestialBodyType",
    "BaseSky",
    "ProceduralSky",
    "HDRISky",
    "StaticSky",
    "CelestialBody",
    "StarField",
    "SkyManager",
    # === Lighting ===
    "DirectionalLight",
    "AmbientLight",
    "SunLight",
    "MoonLight",
    "EnvironmentLighting",
    "LightProbe",
    "LightProbeGrid",
]
