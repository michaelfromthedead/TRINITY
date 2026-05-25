"""Material system constants and configuration values.

This module centralizes magic numbers and configuration constants
for the Materials & Shading subsystem.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


# =============================================================================
# PBR Parameter Ranges
# =============================================================================

@dataclass(frozen=True, slots=True)
class PBRParameterRange:
    """Range definition for a PBR parameter."""
    min_value: float
    max_value: float
    default_value: float


# PBR metallic-roughness workflow parameter ranges
PBR_METALLIC_RANGE = PBRParameterRange(min_value=0.0, max_value=1.0, default_value=0.0)
PBR_ROUGHNESS_RANGE = PBRParameterRange(min_value=0.0, max_value=1.0, default_value=0.5)
PBR_NORMAL_SCALE_RANGE = PBRParameterRange(min_value=0.0, max_value=2.0, default_value=1.0)
PBR_AO_RANGE = PBRParameterRange(min_value=0.0, max_value=1.0, default_value=1.0)
PBR_BASE_COLOR_RANGE = PBRParameterRange(min_value=0.0, max_value=1.0, default_value=1.0)

# Emissive has no upper bound
PBR_EMISSIVE_MIN = 0.0

# Clear coat ranges
CLEAR_COAT_INTENSITY_RANGE = PBRParameterRange(min_value=0.0, max_value=1.0, default_value=1.0)
CLEAR_COAT_ROUGHNESS_RANGE = PBRParameterRange(min_value=0.0, max_value=1.0, default_value=0.0)
CLEAR_COAT_IOR_RANGE = PBRParameterRange(min_value=1.0, max_value=3.0, default_value=1.5)

# Anisotropy ranges
ANISOTROPY_STRENGTH_RANGE = PBRParameterRange(min_value=-1.0, max_value=1.0, default_value=0.0)

# Sheen ranges
SHEEN_ROUGHNESS_RANGE = PBRParameterRange(min_value=0.0, max_value=1.0, default_value=0.5)
SHEEN_INTENSITY_RANGE = PBRParameterRange(min_value=0.0, max_value=2.0, default_value=1.0)

# Iridescence ranges
IRIDESCENCE_INTENSITY_RANGE = PBRParameterRange(min_value=0.0, max_value=1.0, default_value=1.0)
IRIDESCENCE_IOR_RANGE = PBRParameterRange(min_value=1.0, max_value=3.0, default_value=1.3)
IRIDESCENCE_THICKNESS_MIN_DEFAULT = 100.0  # nanometers
IRIDESCENCE_THICKNESS_MAX_DEFAULT = 400.0  # nanometers

# Transmission ranges
TRANSMISSION_FACTOR_RANGE = PBRParameterRange(min_value=0.0, max_value=1.0, default_value=1.0)
TRANSMISSION_IOR_RANGE = PBRParameterRange(min_value=1.0, max_value=3.0, default_value=1.5)
TRANSMISSION_ROUGHNESS_RANGE = PBRParameterRange(min_value=0.0, max_value=1.0, default_value=0.0)


# =============================================================================
# Shader Compilation Constants
# =============================================================================

# Default compilation timeout in seconds
SHADER_COMPILATION_TIMEOUT_SECONDS = 30.0

# Maximum compilation timeout (prevents runaway compilations)
SHADER_COMPILATION_MAX_TIMEOUT_SECONDS = 120.0

# Hot-reload file watcher poll interval in seconds
HOT_RELOAD_POLL_INTERVAL_SECONDS = 0.5

# Content hash truncation length (for cache keys)
SHADER_HASH_LENGTH = 16


# =============================================================================
# PSO Cache Constants
# =============================================================================

# Default maximum number of cached Pipeline State Objects
PSO_CACHE_DEFAULT_MAX_SIZE = 1024

# Minimum PSO cache size
PSO_CACHE_MIN_SIZE = 64

# Maximum PSO cache size (to prevent memory issues)
PSO_CACHE_MAX_SIZE = 16384


# =============================================================================
# Material Graph Constants
# =============================================================================

# Maximum number of nodes in a material graph
MATERIAL_GRAPH_MAX_NODES = 1024

# Maximum depth of node connections (to detect deep recursion)
MATERIAL_GRAPH_MAX_DEPTH = 128

# Default UV coordinate set index
DEFAULT_UV_INDEX = 0


# =============================================================================
# Subsurface Scattering Constants
# =============================================================================

# Default number of diffusion profile samples
SSS_DEFAULT_PROFILE_SAMPLES = 16

# Burley diffusion profile coefficient A
SSS_BURLEY_COEFFICIENT_A = 1.0 / (2.0 * 3.14159265359)

# Burley diffusion profile coefficient B divisor
SSS_BURLEY_COEFFICIENT_B_DIVISOR = 6.0


# =============================================================================
# Color Space Constants
# =============================================================================

# sRGB gamma value
SRGB_GAMMA = 2.2

# Inverse sRGB gamma
SRGB_GAMMA_INVERSE = 1.0 / 2.2

# Rec. 709 luminance coefficients (ITU-R BT.709)
LUMINANCE_COEFFICIENTS_R = 0.2126
LUMINANCE_COEFFICIENTS_G = 0.7152
LUMINANCE_COEFFICIENTS_B = 0.0722


# =============================================================================
# Safe Division Constants
# =============================================================================

# Epsilon for safe division to prevent divide-by-zero
SAFE_DIVISION_EPSILON = 0.0001


__all__ = [
    # Classes
    "PBRParameterRange",
    # PBR ranges
    "PBR_METALLIC_RANGE",
    "PBR_ROUGHNESS_RANGE",
    "PBR_NORMAL_SCALE_RANGE",
    "PBR_AO_RANGE",
    "PBR_BASE_COLOR_RANGE",
    "PBR_EMISSIVE_MIN",
    # Clear coat
    "CLEAR_COAT_INTENSITY_RANGE",
    "CLEAR_COAT_ROUGHNESS_RANGE",
    "CLEAR_COAT_IOR_RANGE",
    # Anisotropy
    "ANISOTROPY_STRENGTH_RANGE",
    # Sheen
    "SHEEN_ROUGHNESS_RANGE",
    "SHEEN_INTENSITY_RANGE",
    # Iridescence
    "IRIDESCENCE_INTENSITY_RANGE",
    "IRIDESCENCE_IOR_RANGE",
    "IRIDESCENCE_THICKNESS_MIN_DEFAULT",
    "IRIDESCENCE_THICKNESS_MAX_DEFAULT",
    # Transmission
    "TRANSMISSION_FACTOR_RANGE",
    "TRANSMISSION_IOR_RANGE",
    "TRANSMISSION_ROUGHNESS_RANGE",
    # Shader compilation
    "SHADER_COMPILATION_TIMEOUT_SECONDS",
    "SHADER_COMPILATION_MAX_TIMEOUT_SECONDS",
    "HOT_RELOAD_POLL_INTERVAL_SECONDS",
    "SHADER_HASH_LENGTH",
    # PSO cache
    "PSO_CACHE_DEFAULT_MAX_SIZE",
    "PSO_CACHE_MIN_SIZE",
    "PSO_CACHE_MAX_SIZE",
    # Material graph
    "MATERIAL_GRAPH_MAX_NODES",
    "MATERIAL_GRAPH_MAX_DEPTH",
    "DEFAULT_UV_INDEX",
    # SSS
    "SSS_DEFAULT_PROFILE_SAMPLES",
    "SSS_BURLEY_COEFFICIENT_A",
    "SSS_BURLEY_COEFFICIENT_B_DIVISOR",
    # Color space
    "SRGB_GAMMA",
    "SRGB_GAMMA_INVERSE",
    "LUMINANCE_COEFFICIENTS_R",
    "LUMINANCE_COEFFICIENTS_G",
    "LUMINANCE_COEFFICIENTS_B",
    # Safe division
    "SAFE_DIVISION_EPSILON",
]
