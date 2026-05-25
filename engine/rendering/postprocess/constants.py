"""
Post-Processing Constants

Centralized configuration values for post-processing effects.
These values should be used instead of magic numbers throughout the codebase.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


# ==============================================================================
# NUMERICAL SAFETY
# ==============================================================================

EPSILON = 1e-6  # General purpose epsilon for avoiding division by zero
SAFE_LOG_MIN = 1e-10  # Minimum value for safe log operations
LUMINANCE_MIN = 1e-6  # Minimum luminance to avoid log(0)


# ==============================================================================
# EXPOSURE CONSTANTS
# ==============================================================================

@dataclass(frozen=True)
class ExposureConstants:
    """Constants for exposure calculations."""

    # EV range limits
    EV_MIN_FALLBACK: float = -10.0  # Fallback EV for invalid inputs
    EV_DEFAULT_MIN: float = -4.0  # Default minimum EV
    EV_DEFAULT_MAX: float = 16.0  # Default maximum EV
    EV_MANUAL_RANGE: Tuple[float, float] = (-10.0, 10.0)

    # Luminance conversion constants
    # Based on ISO 12232:2006 saturation-based speed
    LUMINANCE_TO_EV_SCALE: float = 100.0 / 12.5  # = 8.0
    MIDDLE_GRAY_LUMINANCE: float = 0.18

    # Adaptation speeds (seconds)
    ADAPTATION_SPEED_UP_DEFAULT: float = 3.0
    ADAPTATION_SPEED_DOWN_DEFAULT: float = 1.0

    # Histogram settings
    HISTOGRAM_BINS_DEFAULT: int = 64
    HISTOGRAM_BINS_MIN: int = 16
    HISTOGRAM_BINS_MAX: int = 256

    # Percentile defaults
    LOW_PERCENTILE_DEFAULT: float = 0.5
    HIGH_PERCENTILE_DEFAULT: float = 0.95


EXPOSURE = ExposureConstants()


# ==============================================================================
# BLOOM CONSTANTS
# ==============================================================================

@dataclass(frozen=True)
class BloomConstants:
    """Constants for bloom effect."""

    # Threshold settings
    THRESHOLD_DEFAULT: float = 1.0
    THRESHOLD_SOFTNESS_DEFAULT: float = 0.5
    CLAMP_MAX_DEFAULT: float = 65504.0  # Half-float max to prevent fireflies
    CLAMP_MAX_SAFE: float = 65000.0  # Slightly below max for safety

    # Mip chain settings
    MIP_COUNT_LOW: int = 3
    MIP_COUNT_MEDIUM: int = 5
    MIP_COUNT_HIGH: int = 6
    MIP_COUNT_ULTRA: int = 8
    MIP_COUNT_MAX: int = 12

    # Blur settings
    GAUSSIAN_RADIUS_DEFAULT: int = 4
    GAUSSIAN_SIGMA_DEFAULT: float = 2.0
    BLUR_ITERATIONS_DEFAULT: int = 2

    # Resolution settings
    RESOLUTION_SCALE_DEFAULT: float = 0.5
    RESOLUTION_SCALE_MIN: float = 0.25
    RESOLUTION_SCALE_MAX: float = 1.0

    # Scatter settings
    SCATTER_DEFAULT: float = 0.7
    SCATTER_MIN: float = 0.0
    SCATTER_MAX: float = 1.0

    # Intensity settings
    INTENSITY_DEFAULT: float = 1.0
    INTENSITY_MAX: float = 10.0


BLOOM = BloomConstants()


# ==============================================================================
# TONEMAPPING CONSTANTS
# ==============================================================================

@dataclass(frozen=True)
class TonemapConstants:
    """Constants for tone mapping operators."""

    # General settings
    WHITE_POINT_DEFAULT: float = 11.2
    WHITE_POINT_MIN: float = 1.0
    WHITE_POINT_MAX: float = 100.0
    GAMMA_DEFAULT: float = 2.2
    SATURATION_DEFAULT: float = 1.0

    # ACES constants (from Academy specifications)
    ACES_INPUT_SCALE_DEFAULT: float = 0.6
    ACES_OUTPUT_SCALE_DEFAULT: float = 1.0

    # ACES fitted curve coefficients (Stephen Hill)
    ACES_A: float = 2.51
    ACES_B: float = 0.03
    ACES_C: float = 2.43
    ACES_D: float = 0.59
    ACES_E: float = 0.14

    # AgX constants
    AGX_MIN_EV: float = -12.47393
    AGX_MAX_EV: float = 4.026069

    # Filmic curve defaults (Uncharted 2 / Hable)
    FILMIC_SHOULDER_STRENGTH: float = 0.22
    FILMIC_LINEAR_STRENGTH: float = 0.30
    FILMIC_LINEAR_ANGLE: float = 0.10
    FILMIC_TOE_STRENGTH: float = 0.20
    FILMIC_TOE_NUMERATOR: float = 0.01
    FILMIC_TOE_DENOMINATOR: float = 0.30


TONEMAP = TonemapConstants()


# ==============================================================================
# DEPTH OF FIELD CONSTANTS
# ==============================================================================

@dataclass(frozen=True)
class DOFConstants:
    """Constants for depth of field effect."""

    # Aperture settings (f-stop)
    APERTURE_DEFAULT: float = 2.8
    APERTURE_MIN: float = 1.0  # f/1.0
    APERTURE_MAX: float = 22.0  # f/22

    # Focal length (mm)
    FOCAL_LENGTH_DEFAULT: float = 50.0
    FOCAL_LENGTH_MIN: float = 8.0  # Ultra-wide
    FOCAL_LENGTH_MAX: float = 500.0  # Telephoto

    # Sensor sizes (mm)
    SENSOR_FULL_FRAME: float = 36.0
    SENSOR_APSC: float = 23.6
    SENSOR_MFT: float = 17.3

    # Focus distance (meters)
    FOCUS_DISTANCE_DEFAULT: float = 5.0
    FOCUS_DISTANCE_MIN: float = 0.1
    FOCUS_DISTANCE_MAX: float = 10000.0

    # Blur radius (pixels)
    MAX_BLUR_RADIUS_DEFAULT: float = 32.0
    MAX_BLUR_RADIUS_MIN: float = 1.0
    MAX_BLUR_RADIUS_MAX: float = 128.0

    # Bokeh settings
    BLADE_COUNT_DEFAULT: int = 6
    BLADE_COUNT_MIN: int = 3
    BLADE_COUNT_MAX: int = 16


DOF = DOFConstants()


# ==============================================================================
# AMBIENT OCCLUSION CONSTANTS
# ==============================================================================

@dataclass(frozen=True)
class AOConstants:
    """Constants for ambient occlusion."""

    # Radius settings (world-space meters)
    RADIUS_DEFAULT: float = 0.5
    RADIUS_MIN: float = 0.01
    RADIUS_MAX: float = 5.0

    # Sample counts
    SAMPLE_COUNT_LOW: int = 4
    SAMPLE_COUNT_MEDIUM: int = 8
    SAMPLE_COUNT_HIGH: int = 16
    SAMPLE_COUNT_ULTRA: int = 32

    # Direction counts (HBAO/GTAO)
    DIRECTION_COUNT_DEFAULT: int = 8
    DIRECTION_COUNT_MIN: int = 4
    DIRECTION_COUNT_MAX: int = 16

    # Intensity settings
    INTENSITY_DEFAULT: float = 1.0
    INTENSITY_MIN: float = 0.0
    INTENSITY_MAX: float = 2.0

    # Bias settings
    BIAS_DEFAULT: float = 0.01
    BIAS_MIN: float = 0.001
    BIAS_MAX: float = 0.1

    # Blur settings
    BLUR_SHARPNESS_DEFAULT: float = 8.0
    BLUR_RADIUS_DEFAULT: int = 4

    # Temporal settings
    TEMPORAL_WEIGHT_DEFAULT: float = 0.9
    TEMPORAL_WEIGHT_MIN: float = 0.0
    TEMPORAL_WEIGHT_MAX: float = 0.99


AO = AOConstants()


# ==============================================================================
# ANTI-ALIASING CONSTANTS
# ==============================================================================

@dataclass(frozen=True)
class AntialiasingConstants:
    """Constants for anti-aliasing."""

    # TAA jitter sample counts
    JITTER_SAMPLES_8: int = 8
    JITTER_SAMPLES_16: int = 16
    JITTER_SAMPLES_32: int = 32

    # TAA history weight
    TAA_HISTORY_WEIGHT_DEFAULT: float = 0.9
    TAA_HISTORY_WEIGHT_MIN: float = 0.8
    TAA_HISTORY_WEIGHT_MAX: float = 0.98

    # TAA sharpening
    TAA_SHARPEN_AMOUNT_DEFAULT: float = 0.25
    TAA_SHARPEN_AMOUNT_MIN: float = 0.0
    TAA_SHARPEN_AMOUNT_MAX: float = 1.0

    # TAA velocity rejection
    TAA_VELOCITY_THRESHOLD_DEFAULT: float = 0.01

    # FXAA thresholds
    FXAA_EDGE_THRESHOLD_DEFAULT: float = 0.166
    FXAA_EDGE_THRESHOLD_MIN_DEFAULT: float = 0.0833
    FXAA_SUBPIXEL_QUALITY_DEFAULT: float = 0.75

    # SMAA settings
    SMAA_THRESHOLD_DEFAULT: float = 0.1
    SMAA_MAX_SEARCH_STEPS_DEFAULT: int = 16
    SMAA_CORNER_ROUNDING_DEFAULT: float = 25.0


AA = AntialiasingConstants()


# ==============================================================================
# MOTION BLUR CONSTANTS
# ==============================================================================

@dataclass(frozen=True)
class MotionBlurConstants:
    """Constants for motion blur."""

    # Sample count
    SAMPLE_COUNT_DEFAULT: int = 16
    SAMPLE_COUNT_MIN: int = 4
    SAMPLE_COUNT_MAX: int = 64

    # Max blur radius (pixels)
    MAX_BLUR_RADIUS_DEFAULT: float = 32.0
    MAX_BLUR_RADIUS_MIN: float = 1.0
    MAX_BLUR_RADIUS_MAX: float = 128.0

    # Tile size for optimization
    TILE_SIZE_DEFAULT: int = 16
    TILE_SIZE_MIN: int = 4
    TILE_SIZE_MAX: int = 64

    # Shutter settings
    SHUTTER_ANGLE_DEFAULT: float = 180.0  # degrees (180 = half frame)
    SHUTTER_ANGLE_MIN: float = 0.0
    SHUTTER_ANGLE_MAX: float = 360.0


MOTION_BLUR = MotionBlurConstants()


# ==============================================================================
# UPSCALING CONSTANTS
# ==============================================================================

@dataclass(frozen=True)
class UpscalingConstants:
    """Constants for super resolution upscaling."""

    # Quality preset scale factors (render scale)
    SCALE_ULTRA_PERFORMANCE: float = 0.33  # ~3x upscale
    SCALE_PERFORMANCE: float = 0.50  # 2x upscale
    SCALE_BALANCED: float = 0.58  # ~1.7x upscale
    SCALE_QUALITY: float = 0.67  # ~1.5x upscale
    SCALE_ULTRA_QUALITY: float = 0.77  # ~1.3x upscale
    SCALE_NATIVE: float = 1.0  # No upscaling

    # Mip bias by quality
    MIP_BIAS_ULTRA_PERFORMANCE: float = -1.58
    MIP_BIAS_PERFORMANCE: float = -1.0
    MIP_BIAS_BALANCED: float = -0.79
    MIP_BIAS_QUALITY: float = -0.58
    MIP_BIAS_ULTRA_QUALITY: float = -0.38
    MIP_BIAS_NATIVE: float = 0.0

    # Sharpening defaults
    SHARPENING_AMOUNT_DEFAULT: float = 0.5
    SHARPENING_AMOUNT_MIN: float = 0.0
    SHARPENING_AMOUNT_MAX: float = 1.0


UPSCALING = UpscalingConstants()


# ==============================================================================
# COLOR GRADING CONSTANTS
# ==============================================================================

@dataclass(frozen=True)
class ColorGradingConstants:
    """Constants for color grading."""

    # White balance temperature range
    TEMPERATURE_MIN: float = -100.0
    TEMPERATURE_MAX: float = 100.0
    TINT_MIN: float = -100.0
    TINT_MAX: float = 100.0

    # Contrast range
    CONTRAST_MIN: float = 0.5
    CONTRAST_MAX: float = 2.0
    CONTRAST_DEFAULT: float = 1.0

    # Saturation range
    SATURATION_MIN: float = 0.0
    SATURATION_MAX: float = 2.0
    SATURATION_DEFAULT: float = 1.0

    # Vibrance range
    VIBRANCE_MIN: float = -1.0
    VIBRANCE_MAX: float = 1.0
    VIBRANCE_DEFAULT: float = 0.0

    # LUT settings
    LUT_SIZE_DEFAULT: int = 32
    LUT_SIZE_MIN: int = 16
    LUT_SIZE_MAX: int = 64


COLOR_GRADING = ColorGradingConstants()


# ==============================================================================
# LUMINANCE COEFFICIENTS
# ==============================================================================

# ITU-R BT.709 luminance coefficients (sRGB)
LUMINANCE_COEFFS_BT709 = (0.2126, 0.7152, 0.0722)

# ITU-R BT.601 luminance coefficients (legacy)
LUMINANCE_COEFFS_BT601 = (0.299, 0.587, 0.114)


def calculate_luminance(
    r: float, g: float, b: float,
    coeffs: Tuple[float, float, float] = LUMINANCE_COEFFS_BT709
) -> float:
    """Calculate luminance from RGB values.

    Args:
        r: Red channel.
        g: Green channel.
        b: Blue channel.
        coeffs: Luminance coefficients.

    Returns:
        Luminance value.
    """
    return coeffs[0] * r + coeffs[1] * g + coeffs[2] * b


__all__ = [
    # Safety constants
    "EPSILON",
    "SAFE_LOG_MIN",
    "LUMINANCE_MIN",
    # Effect constants
    "EXPOSURE",
    "BLOOM",
    "TONEMAP",
    "DOF",
    "AO",
    "AA",
    "MOTION_BLUR",
    "UPSCALING",
    "COLOR_GRADING",
    # Luminance
    "LUMINANCE_COEFFS_BT709",
    "LUMINANCE_COEFFS_BT601",
    "calculate_luminance",
]
