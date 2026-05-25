"""Spatial Audio Configuration Constants.

All constants for the spatial audio subsystem including attenuation,
HRTF, Doppler, occlusion, reverb, and propagation parameters.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Final

# =============================================================================
# Attenuation Constants
# =============================================================================

MIN_ATTENUATION_DISTANCE: Final[float] = 1.0
"""Minimum distance before attenuation begins (meters)."""

MAX_ATTENUATION_DISTANCE: Final[float] = 100.0
"""Maximum audible distance - sound culled beyond this (meters)."""

DEFAULT_ROLLOFF: Final[float] = 1.0
"""Default rolloff factor for attenuation curves."""

CONE_INNER_ANGLE: Final[float] = 45.0
"""Inner cone angle where sound is at full volume (degrees)."""

CONE_OUTER_ANGLE: Final[float] = 90.0
"""Outer cone angle where sound attenuates to outer gain (degrees)."""

CONE_OUTER_GAIN: Final[float] = 0.3
"""Gain multiplier at and beyond outer cone angle (0.0-1.0)."""


class AttenuationModel(Enum):
    """Supported distance attenuation models."""

    LINEAR = auto()
    """Linear falloff: volume = 1 - (distance - min) / (max - min)"""

    LOGARITHMIC = auto()
    """Logarithmic falloff: realistic sound decay."""

    INVERSE = auto()
    """Inverse distance: volume = min_distance / distance"""

    INVERSE_SQUARED = auto()
    """Inverse square law: physically accurate for point sources."""

    CUSTOM = auto()
    """Custom designer-defined falloff curve."""

    NONE = auto()
    """No attenuation - constant volume regardless of distance."""


class AttenuationShape(Enum):
    """Shapes for attenuation volumes."""

    SPHERE = auto()
    """Spherical falloff from point."""

    CONE = auto()
    """Directional cone with inner/outer angles."""

    BOX = auto()
    """Axis-aligned bounding box."""

    CAPSULE = auto()
    """Capsule (cylinder with hemispherical ends)."""

    ORIENTED_BOX = auto()
    """Oriented bounding box with rotation."""


# =============================================================================
# HRTF Constants
# =============================================================================

HRTF_SAMPLE_RATE: Final[int] = 48000
"""Sample rate for HRTF processing (Hz)."""

HRTF_FILTER_LENGTH: Final[int] = 128
"""Length of HRTF filter in samples."""

MAX_ITD_SAMPLES: Final[int] = 44
"""Maximum interaural time difference in samples (~1ms at 48kHz)."""

ILD_MAX_DB: Final[float] = 20.0
"""Maximum interaural level difference in decibels."""

HEAD_RADIUS: Final[float] = 0.0875
"""Average human head radius in meters."""

EAR_OFFSET: Final[float] = 0.09
"""Distance from head center to ear in meters."""

HRTF_AZIMUTH_RESOLUTION: Final[float] = 5.0
"""Angular resolution for HRTF lookup tables (degrees)."""

HRTF_ELEVATION_RESOLUTION: Final[float] = 5.0
"""Elevation resolution for HRTF lookup tables (degrees)."""

HRTF_MIN_ELEVATION: Final[float] = -90.0
"""Minimum elevation angle (degrees, -90 = below)."""

HRTF_MAX_ELEVATION: Final[float] = 90.0
"""Maximum elevation angle (degrees, +90 = above)."""


class HRTFQuality(Enum):
    """Quality levels for HRTF processing."""

    LOW = auto()
    """Fast processing with basic binaural cues."""

    MEDIUM = auto()
    """Balanced quality and performance."""

    HIGH = auto()
    """High-fidelity HRTF with elevation support."""

    PERSONALIZED = auto()
    """Personalized HRTF profile."""


# =============================================================================
# Doppler Effect Constants
# =============================================================================

DOPPLER_FACTOR: Final[float] = 1.0
"""Default Doppler factor (1.0 = realistic, >1 = exaggerated)."""

SPEED_OF_SOUND: Final[float] = 343.0
"""Speed of sound in air at 20C (meters per second)."""

MAX_DOPPLER_SHIFT: Final[float] = 2.0
"""Maximum Doppler pitch shift multiplier to prevent artifacts."""

MIN_DOPPLER_SHIFT: Final[float] = 0.5
"""Minimum Doppler pitch shift multiplier."""

DOPPLER_SMOOTHING_TIME: Final[float] = 0.05
"""Time constant for Doppler smoothing to avoid artifacts (seconds)."""

DOPPLER_VELOCITY_THRESHOLD: Final[float] = 0.1
"""Minimum velocity to apply Doppler effect (m/s)."""


# =============================================================================
# Occlusion Constants
# =============================================================================

OCCLUSION_MAX_RAYS: Final[int] = 8
"""Maximum number of rays for multi-ray occlusion detection."""

OCCLUSION_LOW_PASS_FREQ: Final[float] = 1000.0
"""Low-pass filter cutoff frequency for occluded sounds (Hz)."""

OCCLUSION_VOLUME_REDUCTION_DB: Final[float] = 12.0
"""Volume reduction for fully occluded sounds (dB)."""

OBSTRUCTION_VOLUME_REDUCTION_DB: Final[float] = 6.0
"""Volume reduction for obstructed sounds (direct path blocked) (dB)."""

OCCLUSION_UPDATE_RATE: Final[float] = 20.0
"""How often to update occlusion state (Hz)."""

OCCLUSION_INTERPOLATION_TIME: Final[float] = 0.1
"""Time to interpolate occlusion changes (seconds)."""


class OcclusionMethod(Enum):
    """Methods for detecting sound occlusion."""

    RAYCAST = auto()
    """Single ray from source to listener."""

    MULTI_RAY = auto()
    """Multiple rays for more accurate occlusion."""

    SPHERE_TRACE = auto()
    """Sphere trace for volumetric occlusion."""

    PRECOMPUTED = auto()
    """Precomputed/baked occlusion data."""

    HYBRID = auto()
    """Combination of raycast and precomputed."""

    NONE = auto()
    """No occlusion detection."""


class OcclusionResponse(Enum):
    """Audio response types for occlusion."""

    LOW_PASS = auto()
    """Apply low-pass filter."""

    VOLUME_REDUCTION = auto()
    """Reduce volume."""

    BOTH = auto()
    """Apply both low-pass and volume reduction."""

    CUSTOM = auto()
    """Custom occlusion response curve."""


# =============================================================================
# Reverb Zone Constants
# =============================================================================

REVERB_BLEND_TIME: Final[float] = 0.5
"""Default time to blend between reverb zones (seconds)."""

MAX_REVERB_ZONES: Final[int] = 4
"""Maximum number of simultaneous reverb zones."""

REVERB_MIN_ROOM_SIZE: Final[float] = 1.0
"""Minimum room size parameter."""

REVERB_MAX_ROOM_SIZE: Final[float] = 100.0
"""Maximum room size parameter."""

REVERB_MIN_RT60: Final[float] = 0.1
"""Minimum reverb decay time RT60 (seconds)."""

REVERB_MAX_RT60: Final[float] = 20.0
"""Maximum reverb decay time RT60 (seconds)."""

DEFAULT_REVERB_WET_MIX: Final[float] = 0.3
"""Default wet/dry mix for reverb (0.0-1.0)."""

DEFAULT_REVERB_PREDELAY: Final[float] = 0.02
"""Default pre-delay for reverb (seconds)."""


class ReverbPreset(Enum):
    """Preset reverb environments."""

    NONE = auto()
    """No reverb (dry)."""

    SMALL_ROOM = auto()
    """Small room (bedroom, office)."""

    MEDIUM_ROOM = auto()
    """Medium room (living room, classroom)."""

    LARGE_ROOM = auto()
    """Large room (hall, gymnasium)."""

    BATHROOM = auto()
    """Bathroom with reflective surfaces."""

    CAVE = auto()
    """Cave or underground space."""

    CATHEDRAL = auto()
    """Large cathedral or church."""

    ARENA = auto()
    """Sports arena or stadium."""

    HANGAR = auto()
    """Large industrial hangar."""

    OUTDOOR = auto()
    """Outdoor space with minimal reflections."""

    UNDERWATER = auto()
    """Underwater acoustics."""

    CUSTOM = auto()
    """Custom reverb parameters."""


# =============================================================================
# Propagation Constants
# =============================================================================

MAX_REFLECTION_ORDER: Final[int] = 2
"""Maximum order of reflections to simulate."""

MAX_DIFFRACTION_PATHS: Final[int] = 4
"""Maximum number of diffraction paths per source."""

PROPAGATION_UPDATE_RATE: Final[float] = 10.0
"""How often to update propagation paths (Hz)."""

MIN_REFLECTION_COEFFICIENT: Final[float] = 0.01
"""Minimum reflection coefficient to consider."""

MAX_PROPAGATION_DISTANCE: Final[float] = 200.0
"""Maximum distance for propagation simulation (meters)."""

DIFFRACTION_ANGLE_THRESHOLD: Final[float] = 15.0
"""Minimum angle for diffraction to occur (degrees)."""

UTD_PATH_DECAY_FACTOR: Final[float] = 0.2
"""Decay factor for UTD (Uniform Theory of Diffraction) path difference attenuation."""

SQRT_ONE_HALF: Final[float] = 0.7071067811865476
"""Square root of 1/2 (1/sqrt(2)), used for diagonal directions and -3dB calculations."""

TRANSMISSION_MIN_THICKNESS: Final[float] = 0.01
"""Minimum wall thickness for transmission calculation (meters)."""

TRANSMISSION_MAX_THICKNESS: Final[float] = 1.0
"""Maximum wall thickness for transmission calculation (meters)."""


class PropagationPath(Enum):
    """Types of sound propagation paths."""

    DIRECT = auto()
    """Direct line-of-sight path."""

    REFLECTION = auto()
    """Reflected off a surface."""

    DIFFRACTION = auto()
    """Diffracted around an edge."""

    TRANSMISSION = auto()
    """Transmitted through a surface."""

    COUPLED = auto()
    """Sound coupled between rooms."""


# =============================================================================
# Spatialization Constants
# =============================================================================

DEFAULT_SPREAD: Final[float] = 0.0
"""Default spatial spread (0.0 = point, 1.0 = omnidirectional)."""

DEFAULT_FOCUS: Final[float] = 1.0
"""Default spatial focus (0.0 = diffuse, 1.0 = focused)."""

PANNING_LAW_DB: Final[float] = -3.0
"""Panning law attenuation at center (-3dB for constant power)."""

VBAP_MIN_SPEAKERS: Final[int] = 2
"""Minimum speakers for VBAP."""

VBAP_MAX_SPEAKERS: Final[int] = 128
"""Maximum speakers for VBAP."""

AMBISONICS_MAX_ORDER: Final[int] = 3
"""Maximum ambisonics order supported."""


class SpatializationMethod(Enum):
    """Spatialization algorithms."""

    PANNING = auto()
    """Simple stereo/surround panning."""

    HRTF = auto()
    """Head-Related Transfer Function (binaural)."""

    VBAP = auto()
    """Vector Base Amplitude Panning."""

    AMBISONICS = auto()
    """Ambisonics (spherical harmonics)."""

    OBJECT_BASED = auto()
    """Object-based audio (Atmos-like)."""

    NONE = auto()
    """No spatialization (mono or pre-spatialized)."""


# =============================================================================
# Speaker Configuration Constants
# =============================================================================

class SpeakerLayout(Enum):
    """Standard speaker configurations."""

    MONO = auto()
    """Mono (1.0)."""

    STEREO = auto()
    """Stereo (2.0)."""

    QUAD = auto()
    """Quadraphonic (4.0)."""

    SURROUND_5_1 = auto()
    """5.1 Surround."""

    SURROUND_7_1 = auto()
    """7.1 Surround."""

    ATMOS_5_1_2 = auto()
    """Dolby Atmos 5.1.2."""

    ATMOS_7_1_4 = auto()
    """Dolby Atmos 7.1.4."""

    BINAURAL = auto()
    """Binaural headphones."""

    CUSTOM = auto()
    """Custom speaker arrangement."""


# Speaker angle positions (degrees from front center)
SPEAKER_ANGLES = {
    SpeakerLayout.STEREO: [(-30.0, 0.0), (30.0, 0.0)],  # L, R
    SpeakerLayout.QUAD: [(-45.0, 0.0), (45.0, 0.0), (-135.0, 0.0), (135.0, 0.0)],  # FL, FR, BL, BR
    SpeakerLayout.SURROUND_5_1: [
        (-30.0, 0.0),   # Front Left
        (30.0, 0.0),    # Front Right
        (0.0, 0.0),     # Center
        (0.0, 0.0),     # LFE (virtual position)
        (-110.0, 0.0),  # Surround Left
        (110.0, 0.0),   # Surround Right
    ],
    SpeakerLayout.SURROUND_7_1: [
        (-30.0, 0.0),   # Front Left
        (30.0, 0.0),    # Front Right
        (0.0, 0.0),     # Center
        (0.0, 0.0),     # LFE
        (-90.0, 0.0),   # Side Left
        (90.0, 0.0),    # Side Right
        (-150.0, 0.0),  # Back Left
        (150.0, 0.0),   # Back Right
    ],
    SpeakerLayout.ATMOS_7_1_4: [
        (-30.0, 0.0),   # Front Left
        (30.0, 0.0),    # Front Right
        (0.0, 0.0),     # Center
        (0.0, 0.0),     # LFE
        (-90.0, 0.0),   # Side Left
        (90.0, 0.0),    # Side Right
        (-150.0, 0.0),  # Back Left
        (150.0, 0.0),   # Back Right
        (-45.0, 45.0),  # Top Front Left
        (45.0, 45.0),   # Top Front Right
        (-135.0, 45.0), # Top Back Left
        (135.0, 45.0),  # Top Back Right
    ],
}


# =============================================================================
# Listener Constants
# =============================================================================

MAX_LISTENERS: Final[int] = 4
"""Maximum simultaneous listeners (for split-screen)."""

LISTENER_BLEND_TIME: Final[float] = 0.1
"""Time to blend when switching listeners (seconds)."""


# =============================================================================
# Source Type Constants
# =============================================================================

class SourceType(Enum):
    """Types of spatial audio sources."""

    POINT = auto()
    """Point source at a single location."""

    AREA = auto()
    """Area source (extended 2D region)."""

    LINE = auto()
    """Line source (along a path)."""

    VOLUME = auto()
    """Volume source (3D region)."""


# =============================================================================
# Default Acoustic Materials
# =============================================================================

# Material absorption coefficients by frequency band (125Hz, 250Hz, 500Hz, 1kHz, 2kHz, 4kHz)
DEFAULT_MATERIALS = {
    "concrete": (0.01, 0.01, 0.02, 0.02, 0.02, 0.03),
    "brick": (0.02, 0.02, 0.03, 0.04, 0.05, 0.07),
    "glass": (0.03, 0.03, 0.03, 0.03, 0.02, 0.02),
    "wood": (0.10, 0.07, 0.05, 0.04, 0.04, 0.04),
    "carpet": (0.08, 0.24, 0.57, 0.69, 0.71, 0.73),
    "curtain": (0.05, 0.12, 0.35, 0.48, 0.38, 0.36),
    "metal": (0.02, 0.02, 0.02, 0.02, 0.02, 0.02),
    "water": (0.01, 0.01, 0.01, 0.01, 0.02, 0.02),
    "grass": (0.11, 0.26, 0.60, 0.69, 0.92, 0.99),
    "gravel": (0.25, 0.60, 0.65, 0.70, 0.75, 0.80),
    "acoustic_tile": (0.30, 0.35, 0.50, 0.65, 0.70, 0.65),
    "drywall": (0.29, 0.10, 0.05, 0.04, 0.07, 0.09),
}

# Transmission loss coefficients (dB per meter)
TRANSMISSION_LOSS = {
    "concrete": 40.0,
    "brick": 35.0,
    "glass": 25.0,
    "wood": 20.0,
    "metal": 45.0,
    "drywall": 15.0,
}


# =============================================================================
# Performance Tuning
# =============================================================================

SPATIAL_UPDATE_BUDGET_MS: Final[float] = 2.0
"""Maximum time budget for spatial audio updates per frame (ms)."""

MAX_ACTIVE_SOURCES: Final[int] = 64
"""Maximum simultaneously active spatial sources."""

SOURCE_CULLING_DISTANCE: Final[float] = 150.0
"""Distance beyond which sources are culled (meters)."""

LOD_DISTANCE_NEAR: Final[float] = 10.0
"""Distance threshold for near LOD."""

LOD_DISTANCE_FAR: Final[float] = 50.0
"""Distance threshold for far LOD."""
