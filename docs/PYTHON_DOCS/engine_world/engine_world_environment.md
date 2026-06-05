# Investigation: engine/world/environment

## Summary
The environment system is a comprehensive, production-quality implementation providing sky rendering with procedural atmospheric scattering (Rayleigh/Mie), a full weather state machine with regional zones, astronomical time-of-day with realistic sun position calculations, and dynamic lighting with sun/moon/ambient integration. This is one of the most complete subsystems in the TRINITY engine with 5,063 lines of working Python code across 7 files.

## Files
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `__init__.py` | 205 | REAL | Comprehensive exports of all environment systems |
| `constants.py` | 392 | REAL | All magic numbers centralized with proper documentation |
| `volumes.py` | 1296 | REAL | 17 volume types (physics, gameplay, visual, audio, nav) |
| `weather.py` | 822 | REAL | Full state machine with transitions, regional zones |
| `time_of_day.py` | 760 | REAL | Astronomical sun position, lighting keyframes, presets |
| `sky.py` | 776 | REAL | Procedural atmosphere, HDRI, skybox, star field |
| `lighting.py` | 812 | REAL | Sun/moon lights, ambient, light probes |

## Environment Components

### Sky System
- **ProceduralSky**: Rayleigh/Mie scattering simulation for realistic atmosphere
- **HDRISky**: Environment map-based sky rendering
- **StaticSky**: Traditional cubemap skybox
- **StarField**: Procedural stars with twinkling (5000 stars by default)
- **CelestialBody**: Sun, moon, planets with orbital mechanics

### Weather System
- **WeatherStateMachine**: 9 weather types with validated transitions
- **WeatherParameters**: 14 parameters (precipitation, wind, clouds, fog, temp, etc.)
- **WeatherPreset**: Default presets for all weather types with particle/audio refs
- **RegionalWeather**: Multiple weather zones with priority-based blending
- **WeatherTransition**: Smooth transitions with easing (smoothstep, ease_in/out)

### Time of Day System
- **TimeOfDayController**: Full day cycle with astronomical sun position
- **TODCurve**: Keyframe-based lighting interpolation
- **TODLighting**: Sun color, ambient, sky colors, fog, moon, shadows
- **TimeOfDayPreset**: Realistic, stylized, always_noon, always_night presets
- **SunPosition**: Azimuth/elevation with direction vector computation

### Lighting System
- **SunLight**: Linked to TOD controller, auto-updates direction/color
- **MoonLight**: Phase-based intensity, opposite-sun positioning
- **AmbientLight**: Hemisphere lighting with sky/ground colors
- **DirectionalLight**: Shadow cascades, bias, softness settings
- **LightProbe**: Spherical harmonics for indirect lighting
- **LightProbeGrid**: Spatial grid for blended indirect lighting

### Volume System (17 types)
- **Physics**: PhysicsVolume, WaterVolume, PainVolume, KillVolume
- **Gameplay**: TriggerVolume, BlockingVolume, CameraVolume, SpawnVolume
- **Visual**: PostProcessVolume, FogVolume, ReflectionVolume, LightmassVolume
- **Audio**: ReverbVolume, AmbientVolume
- **Navigation**: NavModifierVolume, NavLinkVolume, NavExcludeVolume
- **VolumeManager**: Efficient lookup, priority blending, spatial queries

## Implementation

### Real sky system? YES
- Full Rayleigh scattering (blue sky)
- Mie scattering (sun glow, haze) with Henyey-Greenstein phase function
- Ozone absorption layer
- Sun disk rendering with corona
- Aerial perspective fog
- Sunrise/sunset horizon glow
- Star field with twinkling at night
- Moon phase support

### Real weather? YES
- State machine with 9 weather types (clear, cloudy, rain, storm, fog, snow, hail, sandstorm, overcast)
- Validated transitions (e.g., clear -> cloudy -> rain -> storm)
- Parameter interpolation during transitions with easing
- Regional weather zones with priority-based blending
- Weather affects lighting (cloud dimming, fog tinting, overcast ambient boost)
- Integration with particle systems and audio (refs to particles/rain, sounds/storm_loop, etc.)

### Real time of day? YES
- Astronomical sun position using solar declination formula
- Latitude/day-of-year seasonal variation
- 8 time periods (night, dawn, sunrise, morning, noon, afternoon, sunset, dusk)
- Keyframe-based lighting curves with smoothstep interpolation
- Time scale support (realtime or accelerated)
- Period change callbacks for gameplay events
- Golden hour detection

## Verdict
**REAL IMPLEMENTATION** - This is a production-quality environment system with comprehensive weather simulation, astronomical time-of-day, physically-based sky rendering, and a robust volume system. All components have real logic, proper interpolation, and are designed to integrate with rendering and gameplay systems.

## Evidence

### Procedural Sky Scattering (sky.py:151-215)
```python
def compute_sky_color(self, view_direction: Direction3) -> Color3:
    """Uses simplified Rayleigh/Mie scattering approximation."""
    sun_dir = self.sun_position.direction
    dot = (view_direction[0] * sun_dir[0] + ...)
    sun_angle = math.acos(dot)

    # Rayleigh scattering (blue sky)
    rayleigh_factor = 1.0 - abs(dot)
    rayleigh_r = self.atmosphere.rayleigh_color[0] * rayleigh_factor * self.atmosphere.rayleigh_density
    
    # Mie scattering (sun glow) - Henyey-Greenstein phase function
    g = self.atmosphere.mie_anisotropy
    mie_phase = (1 - g * g) / (4 * math.pi * pow(1 + g * g - 2 * g * dot, 1.5) + 0.001)
```

### Weather State Machine Transitions (weather.py:361-371)
```python
DEFAULT_TRANSITIONS: Dict[WeatherType, Set[WeatherType]] = {
    WeatherType.CLEAR: {WeatherType.CLOUDY, WeatherType.FOG, WeatherType.OVERCAST},
    WeatherType.CLOUDY: {WeatherType.CLEAR, WeatherType.RAIN, WeatherType.OVERCAST, WeatherType.FOG},
    WeatherType.RAIN: {WeatherType.CLOUDY, WeatherType.STORM, WeatherType.FOG},
    WeatherType.STORM: {WeatherType.RAIN, WeatherType.HAIL},
    # ... valid transitions between weather states
}
```

### Astronomical Sun Position (time_of_day.py:662-704)
```python
def get_sun_position(self) -> SunPosition:
    """Calculate sun position based on time, latitude, and day of year."""
    hour_angle = (solar_time - 12.0) * 15.0  # degrees
    
    # Solar declination (seasonal variation)
    day_angle = 2.0 * math.pi * (self._day_of_year - 1) / 365.0
    declination = math.radians(
        23.45 * math.sin(day_angle + math.pi / 2 - 2 * math.pi * 81 / 365)
    )
    
    # Solar elevation angle
    sin_elevation = (
        math.sin(lat_rad) * math.sin(declination) +
        math.cos(lat_rad) * math.cos(declination) * math.cos(math.radians(hour_angle))
    )
```

### Volume Priority Blending (volumes.py:1175-1225)
```python
def get_blended_settings(self, point: Point3D, volume_type: VolumeType) -> Dict[str, Any]:
    """Get blended settings from all volumes of a type at a point."""
    volumes = self.get_volumes_at_point(point, volume_type)
    # Calculate blend weights based on priority and edge distance
    for volume in volumes:
        weight = volume.get_blend_weight(point) * (volume.priority + 1)
        weights.append(weight)
    # Normalize and blend numeric/color values
```

## Integration Points
- Weather parameters feed into lighting system (cloud dimming, fog)
- TOD controller drives sun/moon positions and lighting
- Sky manager integrates with TOD for sun position updates
- Volumes integrate with physics, audio, and visual systems
- All systems support callbacks for gameplay integration
