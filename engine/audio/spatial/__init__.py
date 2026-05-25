"""Spatial Audio Subsystem.

Provides comprehensive 3D audio positioning, attenuation, spatialization,
HRTF, Doppler effect, occlusion, propagation, and acoustic materials.

Modules:
    config: Configuration constants and enums
    positioning: Audio source positioning (point, area, line, volume)
    attenuation: Distance-based volume falloff
    spatialization: 3D audio rendering (panning, HRTF, VBAP, Ambisonics)
    hrtf: Head-Related Transfer Function processing
    doppler: Doppler effect calculation
    speaker_config: Speaker layout configurations
    reverb_zone: Environmental reverb zones
    occlusion: Audio occlusion and obstruction detection
    propagation: Sound propagation paths (direct, reflections, diffraction)
    materials: Acoustic material properties
"""

from __future__ import annotations

# Configuration and constants
from engine.audio.spatial.config import (
    # Attenuation
    AttenuationModel,
    AttenuationShape,
    CONE_INNER_ANGLE,
    CONE_OUTER_ANGLE,
    CONE_OUTER_GAIN,
    DEFAULT_ROLLOFF,
    MAX_ATTENUATION_DISTANCE,
    MIN_ATTENUATION_DISTANCE,
    # HRTF
    EAR_OFFSET,
    HEAD_RADIUS,
    HRTF_AZIMUTH_RESOLUTION,
    HRTF_ELEVATION_RESOLUTION,
    HRTF_FILTER_LENGTH,
    HRTF_MAX_ELEVATION,
    HRTF_MIN_ELEVATION,
    HRTF_SAMPLE_RATE,
    HRTFQuality,
    ILD_MAX_DB,
    MAX_ITD_SAMPLES,
    # Doppler
    DOPPLER_FACTOR,
    DOPPLER_SMOOTHING_TIME,
    DOPPLER_VELOCITY_THRESHOLD,
    MAX_DOPPLER_SHIFT,
    MIN_DOPPLER_SHIFT,
    SPEED_OF_SOUND,
    # Occlusion
    OCCLUSION_INTERPOLATION_TIME,
    OCCLUSION_LOW_PASS_FREQ,
    OCCLUSION_MAX_RAYS,
    OCCLUSION_UPDATE_RATE,
    OCCLUSION_VOLUME_REDUCTION_DB,
    OBSTRUCTION_VOLUME_REDUCTION_DB,
    OcclusionMethod,
    OcclusionResponse,
    # Reverb
    DEFAULT_REVERB_PREDELAY,
    DEFAULT_REVERB_WET_MIX,
    MAX_REVERB_ZONES,
    REVERB_BLEND_TIME,
    REVERB_MAX_ROOM_SIZE,
    REVERB_MAX_RT60,
    REVERB_MIN_ROOM_SIZE,
    REVERB_MIN_RT60,
    ReverbPreset,
    # Propagation
    DIFFRACTION_ANGLE_THRESHOLD,
    MAX_DIFFRACTION_PATHS,
    MAX_PROPAGATION_DISTANCE,
    MAX_REFLECTION_ORDER,
    MIN_REFLECTION_COEFFICIENT,
    PROPAGATION_UPDATE_RATE,
    PropagationPath as PropagationPathType,
    TRANSMISSION_MAX_THICKNESS,
    TRANSMISSION_MIN_THICKNESS,
    # Spatialization
    AMBISONICS_MAX_ORDER,
    DEFAULT_FOCUS,
    DEFAULT_SPREAD,
    PANNING_LAW_DB,
    SpatializationMethod,
    VBAP_MAX_SPEAKERS,
    VBAP_MIN_SPEAKERS,
    # Speaker config
    SPEAKER_ANGLES,
    SpeakerLayout,
    # Source types
    SourceType,
    # Listener
    LISTENER_BLEND_TIME,
    MAX_LISTENERS,
    # Materials
    DEFAULT_MATERIALS,
    TRANSMISSION_LOSS,
    # Performance
    LOD_DISTANCE_FAR,
    LOD_DISTANCE_NEAR,
    MAX_ACTIVE_SOURCES,
    SOURCE_CULLING_DISTANCE,
    SPATIAL_UPDATE_BUDGET_MS,
)

# Positioning
from engine.audio.spatial.positioning import (
    AreaSource,
    create_source,
    LineSource,
    ListenerManager,
    ListenerState,
    PointSource,
    SpatialSource,
    SpatialSourceState,
    VolumeSource,
)

# Attenuation
from engine.audio.spatial.attenuation import (
    AttenuationCurve,
    AttenuationVolume,
    ConeAttenuation,
    create_attenuation,
    CurvePoint,
    CustomCurveAttenuation,
    InverseAttenuation,
    InverseSquaredAttenuation,
    LinearAttenuation,
    LogarithmicAttenuation,
    NoAttenuation,
    get_preset as get_attenuation_preset,
)

# Spatialization
from engine.audio.spatial.spatialization import (
    AmbisonicsSpatializer,
    ChannelGains,
    create_spatializer,
    NoSpatializer,
    spatialize,
    SpatializationParams,
    SpatializationResult,
    Spatializer,
    StereoPanner,
    SurroundPanner,
    VBAPSpatializer,
)

# HRTF
from engine.audio.spatial.hrtf import (
    calculate_ild,
    calculate_itd,
    create_default_hrtf_profile,
    HRTFCoefficients,
    HRTFProcessingState,
    HRTFProfile,
    HRTFSpatializer,
    process_hrtf_block,
)

# Doppler
from engine.audio.spatial.doppler import (
    calculate_doppler_shift,
    DopplerConfig,
    DopplerProcessor,
    DopplerState,
    DOPPLER_PRESETS,
    estimate_arrival_time,
    get_doppler_preset,
)

# Speaker configuration
from engine.audio.spatial.speaker_config import (
    apply_mix_matrix,
    CHANNEL_INDICES,
    ChannelName,
    ChannelRouter,
    create_speaker_config,
    get_channel_count,
    SpeakerConfiguration,
    SpeakerPosition,
    VirtualSpeaker,
)

# Reverb zones
from engine.audio.spatial.reverb_zone import (
    create_reverb_zone,
    get_preset_parameters,
    REVERB_PRESET_PARAMS,
    ReverbParameters,
    ReverbZone,
    ReverbZoneManager,
    ReverbZoneState,
)

# Occlusion
from engine.audio.spatial.occlusion import (
    db_to_linear,
    linear_to_db,
    OcclusionDetector,
    OcclusionProcessor,
    OcclusionResult,
    OcclusionSettings,
    OcclusionState,
    OcclusionType,
    RaycastFunction,
    RaycastHit,
)

# Propagation
from engine.audio.spatial.propagation import (
    DiffractionEdge,
    EdgeQueryFunction,
    GeometryRaycastFunction,
    PathType,
    PropagationCache,
    PropagationCalculator,
    PropagationPath,
    PropagationResult,
    PropagationSettings,
    ReflectionSurface,
    SurfaceQueryFunction,
)

# Materials
from engine.audio.spatial.materials import (
    AcousticMaterial,
    calculate_absorption_area,
    create_custom_material,
    FREQUENCY_BANDS,
    get_transmission_loss_db,
    MaterialDatabase,
    MaterialType,
    MATERIAL_PRESETS,
)


__all__ = [
    # Config constants and enums
    "AttenuationModel",
    "AttenuationShape",
    "CONE_INNER_ANGLE",
    "CONE_OUTER_ANGLE",
    "CONE_OUTER_GAIN",
    "DEFAULT_ROLLOFF",
    "MAX_ATTENUATION_DISTANCE",
    "MIN_ATTENUATION_DISTANCE",
    "EAR_OFFSET",
    "HEAD_RADIUS",
    "HRTF_AZIMUTH_RESOLUTION",
    "HRTF_ELEVATION_RESOLUTION",
    "HRTF_FILTER_LENGTH",
    "HRTF_MAX_ELEVATION",
    "HRTF_MIN_ELEVATION",
    "HRTF_SAMPLE_RATE",
    "HRTFQuality",
    "ILD_MAX_DB",
    "MAX_ITD_SAMPLES",
    "DOPPLER_FACTOR",
    "DOPPLER_SMOOTHING_TIME",
    "DOPPLER_VELOCITY_THRESHOLD",
    "MAX_DOPPLER_SHIFT",
    "MIN_DOPPLER_SHIFT",
    "SPEED_OF_SOUND",
    "OCCLUSION_INTERPOLATION_TIME",
    "OCCLUSION_LOW_PASS_FREQ",
    "OCCLUSION_MAX_RAYS",
    "OCCLUSION_UPDATE_RATE",
    "OCCLUSION_VOLUME_REDUCTION_DB",
    "OBSTRUCTION_VOLUME_REDUCTION_DB",
    "OcclusionMethod",
    "OcclusionResponse",
    "DEFAULT_REVERB_PREDELAY",
    "DEFAULT_REVERB_WET_MIX",
    "MAX_REVERB_ZONES",
    "REVERB_BLEND_TIME",
    "REVERB_MAX_ROOM_SIZE",
    "REVERB_MAX_RT60",
    "REVERB_MIN_ROOM_SIZE",
    "REVERB_MIN_RT60",
    "ReverbPreset",
    "DIFFRACTION_ANGLE_THRESHOLD",
    "MAX_DIFFRACTION_PATHS",
    "MAX_PROPAGATION_DISTANCE",
    "MAX_REFLECTION_ORDER",
    "MIN_REFLECTION_COEFFICIENT",
    "PROPAGATION_UPDATE_RATE",
    "PropagationPathType",
    "TRANSMISSION_MAX_THICKNESS",
    "TRANSMISSION_MIN_THICKNESS",
    "AMBISONICS_MAX_ORDER",
    "DEFAULT_FOCUS",
    "DEFAULT_SPREAD",
    "PANNING_LAW_DB",
    "SpatializationMethod",
    "VBAP_MAX_SPEAKERS",
    "VBAP_MIN_SPEAKERS",
    "SPEAKER_ANGLES",
    "SpeakerLayout",
    "SourceType",
    "LISTENER_BLEND_TIME",
    "MAX_LISTENERS",
    "DEFAULT_MATERIALS",
    "TRANSMISSION_LOSS",
    "LOD_DISTANCE_FAR",
    "LOD_DISTANCE_NEAR",
    "MAX_ACTIVE_SOURCES",
    "SOURCE_CULLING_DISTANCE",
    "SPATIAL_UPDATE_BUDGET_MS",
    # Positioning
    "SpatialSource",
    "ListenerState",
    "ListenerManager",
    "PointSource",
    "AreaSource",
    "LineSource",
    "VolumeSource",
    "SpatialSourceState",
    "create_source",
    # Attenuation
    "AttenuationCurve",
    "LinearAttenuation",
    "LogarithmicAttenuation",
    "InverseAttenuation",
    "InverseSquaredAttenuation",
    "NoAttenuation",
    "CustomCurveAttenuation",
    "CurvePoint",
    "ConeAttenuation",
    "AttenuationVolume",
    "create_attenuation",
    "get_attenuation_preset",
    # Spatialization
    "SpatializationParams",
    "ChannelGains",
    "Spatializer",
    "StereoPanner",
    "SurroundPanner",
    "VBAPSpatializer",
    "AmbisonicsSpatializer",
    "NoSpatializer",
    "SpatializationResult",
    "create_spatializer",
    "spatialize",
    # HRTF
    "HRTFCoefficients",
    "HRTFProfile",
    "HRTFSpatializer",
    "HRTFProcessingState",
    "calculate_itd",
    "calculate_ild",
    "create_default_hrtf_profile",
    "process_hrtf_block",
    # Doppler
    "calculate_doppler_shift",
    "DopplerConfig",
    "DopplerProcessor",
    "DopplerState",
    "DOPPLER_PRESETS",
    "estimate_arrival_time",
    "get_doppler_preset",
    # Speaker config
    "ChannelName",
    "CHANNEL_INDICES",
    "SpeakerPosition",
    "SpeakerConfiguration",
    "ChannelRouter",
    "VirtualSpeaker",
    "get_channel_count",
    "apply_mix_matrix",
    "create_speaker_config",
    # Reverb zones
    "ReverbParameters",
    "ReverbZone",
    "ReverbZoneManager",
    "ReverbZoneState",
    "REVERB_PRESET_PARAMS",
    "get_preset_parameters",
    "create_reverb_zone",
    # Occlusion
    "OcclusionType",
    "OcclusionResult",
    "OcclusionDetector",
    "OcclusionProcessor",
    "OcclusionSettings",
    "OcclusionState",
    "RaycastHit",
    "RaycastFunction",
    "db_to_linear",
    "linear_to_db",
    # Propagation
    "PathType",
    "PropagationPath",
    "PropagationResult",
    "PropagationCalculator",
    "PropagationSettings",
    "PropagationCache",
    "ReflectionSurface",
    "DiffractionEdge",
    "GeometryRaycastFunction",
    "EdgeQueryFunction",
    "SurfaceQueryFunction",
    # Materials
    "MaterialType",
    "AcousticMaterial",
    "MaterialDatabase",
    "MATERIAL_PRESETS",
    "FREQUENCY_BANDS",
    "create_custom_material",
    "get_transmission_loss_db",
    "calculate_absorption_area",
]
