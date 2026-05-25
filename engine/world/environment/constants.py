"""
Environment System Constants.

Centralizes all magic numbers and configuration values for the environment
subsystem including volumes, weather, time of day, sky, and lighting.
"""

from typing import Dict, Tuple

# =============================================================================
# Volume System Constants
# =============================================================================

# Default volume parameters
DEFAULT_VOLUME_BLEND_RADIUS: float = 0.0
DEFAULT_VOLUME_PRIORITY: int = 0

# Physics volume defaults
DEFAULT_GRAVITY_STRENGTH: float = 9.81  # m/s^2 (Earth gravity)
DEFAULT_GRAVITY_DIRECTION: Tuple[float, float, float] = (0.0, -1.0, 0.0)

# Water volume defaults
DEFAULT_WATER_BUOYANCY: float = 1.0
DEFAULT_WATER_DRAG: float = 0.5
DEFAULT_WATER_DRAG_MIN: float = 0.0
DEFAULT_WATER_DRAG_MAX: float = 1.0

# Pain volume defaults
DEFAULT_PAIN_DAMAGE_PER_SECOND: float = 10.0

# Post-process volume defaults
DEFAULT_EXPOSURE: float = 1.0
MIN_EXPOSURE: float = 0.01
DEFAULT_SATURATION: float = 1.0
DEFAULT_CONTRAST: float = 1.0
DEFAULT_BLOOM_INTENSITY: float = 0.0
DEFAULT_VIGNETTE_INTENSITY: float = 0.0
MAX_VIGNETTE_INTENSITY: float = 1.0

# Fog volume defaults
DEFAULT_FOG_DENSITY: float = 0.02
DEFAULT_FOG_HEIGHT_FALLOFF: float = 0.2

# Reverb volume defaults
DEFAULT_ROOM_SIZE: float = 0.5
DEFAULT_DECAY_TIME: float = 1.0
DEFAULT_WET_DRY_MIX: float = 0.5

# Camera volume defaults
DEFAULT_CAMERA_BLEND_TIME: float = 1.0

# Reflection volume defaults
DEFAULT_REFLECTION_INFLUENCE_RADIUS: float = 100.0
DEFAULT_REFLECTION_BRIGHTNESS: float = 1.0

# Mathematical constants for bounds
DEGENERATE_SEGMENT_THRESHOLD: float = 1e-10


# =============================================================================
# Weather System Constants
# =============================================================================

# Transition defaults
DEFAULT_WEATHER_TRANSITION_DURATION: float = 60.0  # seconds
DEFAULT_WEATHER_EASING: str = "smoothstep"

# Weather zone defaults
DEFAULT_WEATHER_ZONE_BLEND_RADIUS: float = 100.0  # meters

# Default weather parameters
DEFAULT_WIND_SPEED: float = 5.0  # m/s
DEFAULT_TEMPERATURE: float = 20.0  # Celsius
DEFAULT_HUMIDITY: float = 0.5
DEFAULT_VISIBILITY: float = 10000.0  # meters
DEFAULT_ATMOSPHERIC_PRESSURE: float = 1013.25  # hPa (standard sea level)

# Weather preset values
STORM_WIND_SPEED_THRESHOLD: float = 15.0  # m/s
HEAVY_PRECIPITATION_THRESHOLD: float = 0.5


# =============================================================================
# Time of Day Constants
# =============================================================================

# Period boundaries (hours) - defines start times for each period
# Format: (start_hour, end_hour)
PERIOD_NIGHT_START: float = 21.0
PERIOD_NIGHT_END: float = 4.5
PERIOD_DAWN_START: float = 4.5
PERIOD_DAWN_END: float = 6.0
PERIOD_SUNRISE_START: float = 6.0
PERIOD_SUNRISE_END: float = 7.5
PERIOD_MORNING_START: float = 7.5
PERIOD_MORNING_END: float = 11.0
PERIOD_NOON_START: float = 11.0
PERIOD_NOON_END: float = 13.0
PERIOD_AFTERNOON_START: float = 13.0
PERIOD_AFTERNOON_END: float = 17.0
PERIOD_SUNSET_START: float = 17.0
PERIOD_SUNSET_END: float = 19.5
PERIOD_DUSK_START: float = 19.5
PERIOD_DUSK_END: float = 21.0

# Time calculation constants
HOURS_PER_DAY: float = 24.0
DAYS_PER_YEAR: int = 365
SECONDS_PER_HOUR: float = 3600.0

# Solar calculation constants (astronomical)
SOLAR_DECLINATION_MAX: float = 23.45  # degrees (axial tilt)
SOLAR_DECLINATION_DAY_OFFSET: int = 81  # days offset for declination calculation
DEGREES_PER_HOUR_ANGLE: float = 15.0  # degrees per hour of time

# Default controller values
DEFAULT_TIME_HOURS: float = 12.0
DEFAULT_TIME_SCALE: float = 1.0
DEFAULT_LATITUDE: float = 45.0
DEFAULT_DAY_OF_YEAR: int = 172  # June 21 (summer solstice northern hemisphere)

# Latitude limits
MIN_LATITUDE: float = -90.0
MAX_LATITUDE: float = 90.0

# Golden hour threshold
GOLDEN_HOUR_MAX_ELEVATION: float = 15.0  # degrees


# =============================================================================
# Sky System Constants
# =============================================================================

# Atmosphere physical constants
DEFAULT_ATMOSPHERE_HEIGHT: float = 100000.0  # meters
EARTH_RADIUS: float = 6371000.0  # meters

# Sun/Moon angular sizes (radians)
SUN_ANGULAR_RADIUS: float = 0.00465  # ~0.27 degrees (actual ~0.25-0.27)
MOON_ANGULAR_SIZE_DEGREES: float = 0.52  # degrees

# Default atmosphere scattering
DEFAULT_RAYLEIGH_DENSITY: float = 1.0
DEFAULT_MIE_DENSITY: float = 1.0
DEFAULT_MIE_ANISOTROPY: float = 0.8  # Forward scattering factor
DEFAULT_OZONE_DENSITY: float = 1.0
DEFAULT_SUN_INTENSITY_MULTIPLIER: float = 1.0

# Default colors (normalized RGB)
DEFAULT_RAYLEIGH_COLOR: Tuple[float, float, float] = (0.175, 0.409, 1.0)  # Blue wavelength
DEFAULT_MIE_COLOR: Tuple[float, float, float] = (1.0, 1.0, 1.0)
DEFAULT_OZONE_COLOR: Tuple[float, float, float] = (0.65, 1.0, 0.85)
DEFAULT_GROUND_ALBEDO: Tuple[float, float, float] = (0.3, 0.3, 0.3)

# Sky color computation
SKY_COLOR_HDR_MAX: float = 10.0  # Maximum HDR sky color value
SUN_DISK_CORONA_MULTIPLIER: float = 2.0  # Corona extends to 2x angular radius
SUN_DISK_INTENSITY: float = 100.0  # Base sun disk intensity
SUNSET_ELEVATION_THRESHOLD: float = 15.0  # degrees for horizon glow

# Aerial perspective
AERIAL_PERSPECTIVE_DENSITY_SCALE: float = 0.00001

# Star field defaults
DEFAULT_STAR_COUNT: int = 5000
DEFAULT_STAR_BRIGHTNESS: float = 1.0
DEFAULT_TWINKLE_SPEED: float = 1.0
STAR_VISIBILITY_SUN_ELEVATION_THRESHOLD: float = 5.0  # degrees
STAR_FADE_SUN_ELEVATION_RANGE: float = 15.0  # degrees for fade transition

# Celestial body defaults
DEFAULT_CELESTIAL_ANGULAR_SIZE: float = 0.5  # degrees
DEFAULT_CELESTIAL_ORBITAL_PERIOD: float = 24.0  # hours
MOON_ORBITAL_PERIOD: float = 24.8  # hours (rises ~50 min later each day)
MOON_ORBITAL_OFFSET: float = 12.0  # hours offset from sun

# Below horizon thresholds
SUN_BELOW_HORIZON_DISK_THRESHOLD: float = -5.0  # degrees
CELESTIAL_BELOW_HORIZON_THRESHOLD: float = -0.1


# =============================================================================
# Lighting System Constants
# =============================================================================

# Directional light defaults
DEFAULT_LIGHT_INTENSITY: float = 1.0
DEFAULT_LIGHT_COLOR: Tuple[float, float, float] = (1.0, 1.0, 1.0)
DEFAULT_LIGHT_DIRECTION: Tuple[float, float, float] = (0.0, -1.0, 0.0)

# Shadow defaults
DEFAULT_SHADOW_CASCADE_COUNT: int = 4
DEFAULT_SHADOW_DISTANCE: float = 500.0  # meters
DEFAULT_SHADOW_BIAS: float = 0.001
DEFAULT_SHADOW_SOFTNESS: float = 0.5
MOON_SHADOW_CASCADE_COUNT: int = 2
MOON_SHADOW_SOFTNESS: float = 0.8

# Sun light defaults
DEFAULT_SUN_COLOR: Tuple[float, float, float] = (1.0, 0.95, 0.9)
SUN_LOW_ELEVATION_THRESHOLD: float = 10.0  # degrees for intensity reduction

# Moon light defaults
DEFAULT_MOON_COLOR: Tuple[float, float, float] = (0.6, 0.7, 0.8)
DEFAULT_MOON_INTENSITY: float = 0.2
DEFAULT_MOON_PHASE: float = 0.5  # 0 = new, 0.5 = full
MOON_DAY_VISIBILITY_FACTOR: float = 30.0  # degrees for day visibility reduction

# Ambient light defaults
DEFAULT_AMBIENT_COLOR: Tuple[float, float, float] = (0.5, 0.6, 0.7)
DEFAULT_AMBIENT_INTENSITY: float = 0.3
DEFAULT_GROUND_COLOR: Tuple[float, float, float] = (0.2, 0.15, 0.1)

# Ambient occlusion defaults
DEFAULT_AO_INTENSITY: float = 0.5
DEFAULT_AO_RADIUS: float = 0.5

# Weather lighting modifiers
CLOUD_DIMMING_FACTOR: float = 0.7  # Max dimming from clouds
OVERCAST_AMBIENT_BOOST: float = 1.3  # Ambient increase during overcast
OVERCAST_AMBIENT_MAX: float = 0.5  # Maximum ambient intensity during overcast
OVERCAST_CLOUD_THRESHOLD: float = 0.7  # Cloud density for overcast effects
FOG_AMBIENT_TINT_THRESHOLD: float = 0.3  # Fog density for ambient tinting
FOG_AMBIENT_TINT_FACTOR: float = 0.5  # Max fog tint contribution

# Light probe defaults
DEFAULT_PROBE_RADIUS: float = 10.0  # meters
DEFAULT_PROBE_INTENSITY: float = 1.0

# Light probe grid defaults
DEFAULT_GRID_BOUNDS_MIN: Tuple[float, float, float] = (-100.0, 0.0, -100.0)
DEFAULT_GRID_BOUNDS_MAX: Tuple[float, float, float] = (100.0, 50.0, 100.0)
DEFAULT_GRID_RESOLUTION: Tuple[int, int, int] = (5, 3, 5)
PROBE_RADIUS_SPACING_MULTIPLIER: float = 1.5

# Shadow visibility threshold
SHADOW_LIGHT_INTENSITY_THRESHOLD: float = 0.1


# =============================================================================
# Color Constants
# =============================================================================

# Color component ranges
COLOR_MIN: float = 0.0
COLOR_MAX: float = 1.0

# Default sky colors
DEFAULT_SKY_COLOR_ZENITH: Tuple[float, float, float] = (0.3, 0.5, 0.9)
DEFAULT_SKY_COLOR_HORIZON: Tuple[float, float, float] = (0.7, 0.8, 0.9)
DEFAULT_FOG_COLOR: Tuple[float, float, float] = (0.7, 0.8, 0.9)


__all__ = [
    # Volume constants
    "DEFAULT_VOLUME_BLEND_RADIUS",
    "DEFAULT_VOLUME_PRIORITY",
    "DEFAULT_GRAVITY_STRENGTH",
    "DEFAULT_GRAVITY_DIRECTION",
    "DEFAULT_WATER_BUOYANCY",
    "DEFAULT_WATER_DRAG",
    "DEFAULT_WATER_DRAG_MIN",
    "DEFAULT_WATER_DRAG_MAX",
    "DEFAULT_PAIN_DAMAGE_PER_SECOND",
    "DEFAULT_EXPOSURE",
    "MIN_EXPOSURE",
    "DEFAULT_SATURATION",
    "DEFAULT_CONTRAST",
    "DEFAULT_BLOOM_INTENSITY",
    "DEFAULT_VIGNETTE_INTENSITY",
    "MAX_VIGNETTE_INTENSITY",
    "DEFAULT_FOG_DENSITY",
    "DEFAULT_FOG_HEIGHT_FALLOFF",
    "DEFAULT_ROOM_SIZE",
    "DEFAULT_DECAY_TIME",
    "DEFAULT_WET_DRY_MIX",
    "DEFAULT_CAMERA_BLEND_TIME",
    "DEFAULT_REFLECTION_INFLUENCE_RADIUS",
    "DEFAULT_REFLECTION_BRIGHTNESS",
    "DEGENERATE_SEGMENT_THRESHOLD",
    # Weather constants
    "DEFAULT_WEATHER_TRANSITION_DURATION",
    "DEFAULT_WEATHER_EASING",
    "DEFAULT_WEATHER_ZONE_BLEND_RADIUS",
    "DEFAULT_WIND_SPEED",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_HUMIDITY",
    "DEFAULT_VISIBILITY",
    "DEFAULT_ATMOSPHERIC_PRESSURE",
    "STORM_WIND_SPEED_THRESHOLD",
    "HEAVY_PRECIPITATION_THRESHOLD",
    # Time of day constants
    "PERIOD_NIGHT_START",
    "PERIOD_NIGHT_END",
    "PERIOD_DAWN_START",
    "PERIOD_DAWN_END",
    "PERIOD_SUNRISE_START",
    "PERIOD_SUNRISE_END",
    "PERIOD_MORNING_START",
    "PERIOD_MORNING_END",
    "PERIOD_NOON_START",
    "PERIOD_NOON_END",
    "PERIOD_AFTERNOON_START",
    "PERIOD_AFTERNOON_END",
    "PERIOD_SUNSET_START",
    "PERIOD_SUNSET_END",
    "PERIOD_DUSK_START",
    "PERIOD_DUSK_END",
    "HOURS_PER_DAY",
    "DAYS_PER_YEAR",
    "SECONDS_PER_HOUR",
    "SOLAR_DECLINATION_MAX",
    "SOLAR_DECLINATION_DAY_OFFSET",
    "DEGREES_PER_HOUR_ANGLE",
    "DEFAULT_TIME_HOURS",
    "DEFAULT_TIME_SCALE",
    "DEFAULT_LATITUDE",
    "DEFAULT_DAY_OF_YEAR",
    "MIN_LATITUDE",
    "MAX_LATITUDE",
    "GOLDEN_HOUR_MAX_ELEVATION",
    # Sky constants
    "DEFAULT_ATMOSPHERE_HEIGHT",
    "EARTH_RADIUS",
    "SUN_ANGULAR_RADIUS",
    "MOON_ANGULAR_SIZE_DEGREES",
    "DEFAULT_RAYLEIGH_DENSITY",
    "DEFAULT_MIE_DENSITY",
    "DEFAULT_MIE_ANISOTROPY",
    "DEFAULT_OZONE_DENSITY",
    "DEFAULT_SUN_INTENSITY_MULTIPLIER",
    "DEFAULT_RAYLEIGH_COLOR",
    "DEFAULT_MIE_COLOR",
    "DEFAULT_OZONE_COLOR",
    "DEFAULT_GROUND_ALBEDO",
    "SKY_COLOR_HDR_MAX",
    "SUN_DISK_CORONA_MULTIPLIER",
    "SUN_DISK_INTENSITY",
    "SUNSET_ELEVATION_THRESHOLD",
    "AERIAL_PERSPECTIVE_DENSITY_SCALE",
    "DEFAULT_STAR_COUNT",
    "DEFAULT_STAR_BRIGHTNESS",
    "DEFAULT_TWINKLE_SPEED",
    "STAR_VISIBILITY_SUN_ELEVATION_THRESHOLD",
    "STAR_FADE_SUN_ELEVATION_RANGE",
    "DEFAULT_CELESTIAL_ANGULAR_SIZE",
    "DEFAULT_CELESTIAL_ORBITAL_PERIOD",
    "MOON_ORBITAL_PERIOD",
    "MOON_ORBITAL_OFFSET",
    "SUN_BELOW_HORIZON_DISK_THRESHOLD",
    "CELESTIAL_BELOW_HORIZON_THRESHOLD",
    # Lighting constants
    "DEFAULT_LIGHT_INTENSITY",
    "DEFAULT_LIGHT_COLOR",
    "DEFAULT_LIGHT_DIRECTION",
    "DEFAULT_SHADOW_CASCADE_COUNT",
    "DEFAULT_SHADOW_DISTANCE",
    "DEFAULT_SHADOW_BIAS",
    "DEFAULT_SHADOW_SOFTNESS",
    "MOON_SHADOW_CASCADE_COUNT",
    "MOON_SHADOW_SOFTNESS",
    "DEFAULT_SUN_COLOR",
    "SUN_LOW_ELEVATION_THRESHOLD",
    "DEFAULT_MOON_COLOR",
    "DEFAULT_MOON_INTENSITY",
    "DEFAULT_MOON_PHASE",
    "MOON_DAY_VISIBILITY_FACTOR",
    "DEFAULT_AMBIENT_COLOR",
    "DEFAULT_AMBIENT_INTENSITY",
    "DEFAULT_GROUND_COLOR",
    "DEFAULT_AO_INTENSITY",
    "DEFAULT_AO_RADIUS",
    "CLOUD_DIMMING_FACTOR",
    "OVERCAST_AMBIENT_BOOST",
    "OVERCAST_AMBIENT_MAX",
    "OVERCAST_CLOUD_THRESHOLD",
    "FOG_AMBIENT_TINT_THRESHOLD",
    "FOG_AMBIENT_TINT_FACTOR",
    "DEFAULT_PROBE_RADIUS",
    "DEFAULT_PROBE_INTENSITY",
    "DEFAULT_GRID_BOUNDS_MIN",
    "DEFAULT_GRID_BOUNDS_MAX",
    "DEFAULT_GRID_RESOLUTION",
    "PROBE_RADIUS_SPACING_MULTIPLIER",
    "SHADOW_LIGHT_INTENSITY_THRESHOLD",
    # Color constants
    "COLOR_MIN",
    "COLOR_MAX",
    "DEFAULT_SKY_COLOR_ZENITH",
    "DEFAULT_SKY_COLOR_HORIZON",
    "DEFAULT_FOG_COLOR",
]
