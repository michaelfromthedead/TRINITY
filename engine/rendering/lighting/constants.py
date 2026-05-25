"""Lighting system constants.

Centralizes all magic numbers and configuration defaults for the lighting
and shadows subsystem. This makes tuning and maintenance easier.
"""

from __future__ import annotations


class ShadowConstants:
    """Shadow mapping constants."""

    # Default shadow map resolutions
    DEFAULT_RESOLUTION: int = 2048
    MIN_RESOLUTION: int = 256
    MAX_RESOLUTION: int = 8192

    # Shadow atlas
    DEFAULT_ATLAS_RESOLUTION: int = 4096

    # Bias values to prevent shadow acne
    DEFAULT_DEPTH_BIAS: float = 0.0001
    DEFAULT_SLOPE_BIAS: float = 0.001
    DEFAULT_NORMAL_BIAS: float = 0.02

    # Filtering
    DEFAULT_FILTER_SIZE: int = 3
    DEFAULT_SOFTNESS: float = 1.0

    # Near plane for shadow cameras
    SHADOW_NEAR_PLANE: float = 0.1


class CSMConstants:
    """Cascaded Shadow Map constants."""

    MIN_CASCADE_COUNT: int = 1
    MAX_CASCADE_COUNT: int = 4
    DEFAULT_CASCADE_COUNT: int = 4

    # Logarithmic split scheme blend factor (0=linear, 1=logarithmic)
    CASCADE_LAMBDA: float = 0.75

    # Default cascade distances
    DEFAULT_CASCADE_DISTANCES: list[float] = [10.0, 30.0, 100.0, 500.0]

    # Light space offset for orthographic projection
    LIGHT_SPACE_OFFSET: float = 100.0

    # Cascade blending range
    DEFAULT_CASCADE_BLEND_RANGE: float = 2.0


class FroxelConstants:
    """Froxel grid (clustered lighting) constants."""

    # Default grid dimensions
    DEFAULT_TILES_X: int = 16
    DEFAULT_TILES_Y: int = 9
    DEFAULT_SLICES_Z: int = 24

    # Default screen resolution for tile calculation
    DEFAULT_SCREEN_WIDTH: int = 1920
    DEFAULT_SCREEN_HEIGHT: int = 1080

    # Default frustum bounds
    DEFAULT_NEAR_PLANE: float = 0.1
    DEFAULT_FAR_PLANE: float = 1000.0

    # Minimum cosine value to prevent division by zero
    MIN_COS_VALUE: float = 0.001


class LightConstants:
    """Light source constants."""

    # Sun angular diameter in radians (approximately 0.53 degrees)
    SUN_ANGULAR_DIAMETER: float = 0.00935

    # Light attenuation
    DEFAULT_FALLOFF_EXPONENT: float = 2.0

    # Area light influence radius multiplier
    AREA_LIGHT_INFLUENCE_MULTIPLIER: float = 10.0

    # Maximum light intensity (to prevent overflow in calculations)
    MAX_INTENSITY: float = 1e6

    # Minimum value for cos calculation to prevent division issues
    MIN_COS_ANGLE: float = 0.001


class GIProbeConstants:
    """Global Illumination probe constants."""

    # Default samples for baking
    DEFAULT_BAKE_SAMPLES: int = 64

    # Spherical harmonics
    SH_L2_COEFFICIENTS: int = 27  # 9 per channel * 3 channels

    # Probe grid defaults
    DEFAULT_PROBE_RADIUS: float = 10.0
    DEFAULT_PROBE_GRID_RESOLUTION: tuple[int, int, int] = (8, 4, 8)

    # Blend distance for volumes
    DEFAULT_BLEND_DISTANCE: float = 1.0
    MIN_BLEND_DISTANCE: float = 0.001  # Prevent division by zero

    # Reflection probe
    DEFAULT_REFLECTION_RESOLUTION: int = 256

    # Lightmap defaults
    DEFAULT_LIGHTMAP_SIZE: int = 256


class DDGIConstants:
    """Dynamic Diffuse Global Illumination constants."""

    # Ray tracing
    DEFAULT_RAYS_PER_PROBE: int = 256
    DEFAULT_MAX_RAY_DISTANCE: float = 100.0

    # Octahedral encoding resolution
    DEFAULT_IRRADIANCE_RESOLUTION: int = 8
    DEFAULT_VISIBILITY_RESOLUTION: int = 16

    # Temporal stability
    DEFAULT_HYSTERESIS: float = 0.97
    DEFAULT_DEPTH_SHARPNESS: float = 50.0

    # Bias values
    DEFAULT_NORMAL_BIAS: float = 0.25
    DEFAULT_VIEW_BIAS: float = 0.25

    # Sleeping probe update frequency (in frames)
    SLEEPING_PROBE_UPDATE_INTERVAL: int = 60

    # Minimum variance for Chebyshev test
    MIN_VARIANCE: float = 0.0001


class ShadowFilterConstants:
    """Shadow filtering constants."""

    # PCF
    DEFAULT_PCF_KERNEL_SIZE: int = 3
    DEFAULT_PCF_BIAS: float = 0.001

    # PCSS
    DEFAULT_BLOCKER_SEARCH_SAMPLES: int = 16
    DEFAULT_PCF_SAMPLES: int = 32
    DEFAULT_LIGHT_SIZE: float = 1.0
    DEFAULT_MAX_FILTER_RADIUS: float = 0.1

    # VSM
    DEFAULT_MIN_VARIANCE: float = 0.00001
    DEFAULT_LIGHT_BLEEDING_REDUCTION: float = 0.2

    # ESM
    DEFAULT_ESM_EXPONENT: float = 80.0
    DEFAULT_ESM_BIAS: float = 0.001

    # Contact shadows
    DEFAULT_CONTACT_SHADOW_DISTANCE: float = 0.5
    DEFAULT_CONTACT_SHADOW_STEPS: int = 16
    DEFAULT_CONTACT_SHADOW_THICKNESS: float = 0.01
    DEFAULT_CONTACT_SHADOW_FADE_START: float = 0.3
    DEFAULT_CONTACT_SHADOW_FADE_END: float = 0.5


# Numerical safety constants
EPSILON: float = 1e-6
SMALL_FLOAT: float = 1e-10
