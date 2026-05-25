"""
Post-Processing Subsystem

Provides comprehensive post-processing effects for the rendering pipeline:
- Post-process stack with ordered effect chain
- Post-process volumes for spatial blending
- Exposure control with auto-exposure and eye adaptation
- Bloom with mip chain and lens dirt
- Tone mapping (Reinhard, ACES, AgX, Filmic)
- Color grading with LUT support
- Depth of field with bokeh simulation
- Motion blur (camera and object)
- Ambient occlusion (SSAO, HBAO, GTAO)
- Anti-aliasing (FXAA, SMAA, TAA)
- Super resolution upscaling (FSR, DLSS, XeSS)
"""

# Post-Process Stack
from .postprocess_stack import (
    BlendMode,
    BoxVolumeShape,
    EffectExecutionPath,
    EffectPriority,
    EffectQuality,
    EffectSettings,
    ExecutionFlags,
    get_quality_preset,
    IntermediateTarget,
    IntermediateTargetManager,
    PostProcessContext,
    PostProcessEffect,
    PostProcessStack,
    PostProcessStackConfig,
    PostProcessStackExecutor,
    PostProcessVolume,
    PostProcessVolumeSettings,
    QUALITY_PRESETS,
    QUALITY_PRESET_HIGH,
    QUALITY_PRESET_LOW,
    QUALITY_PRESET_MEDIUM,
    QUALITY_PRESET_ULTRA,
    QualityPreset,
    SphereVolumeShape,
    VolumeShape,
)

# Exposure
from .exposure import (
    AdaptationCurve,
    AutoExposure,
    ev_to_exposure,
    ExposureCalculator,
    ExposureEffect,
    ExposureMode,
    exposure_to_ev,
    ExposureSettings,
    EyeAdaptation,
    HistogramExposure,
    luminance_to_ev,
    ManualExposure,
    MeteringMode,
)

# Bloom
from .bloom import (
    BloomBlur,
    BloomDownsample,
    BloomEffect,
    BloomMipSettings,
    BloomQuality,
    BloomSettings,
    BloomThreshold,
    BloomUpsample,
    BlurMethod,
    LensDirtSettings,
)

# Tone Mapping
from .tonemapping import (
    ACES,
    ACESFitted,
    AgX,
    CustomCurve,
    CustomCurveSettings,
    Filmic,
    Reinhard,
    ReinhardExtended,
    TonemapCurvePoint,
    TonemapFunction,
    TonemapOperator,
    TonemappingEffect,
    TonemapSettings,
)

# Color Grading
from .color_grading import (
    ColorGradingEffect,
    ColorGradingSettings,
    ColorGradingStack,
    ColorSpace,
    ContrastSettings,
    HueSatLightness,
    LiftGammaGain,
    LUT3D,
    LUT3DSettings,
    LUTFormat,
    SaturationSettings,
    WhiteBalanceSettings,
)

# Depth of Field
from .dof import (
    AutoFocusSystem,
    BokehShape,
    BokehShapeType,
    CircleOfConfusion,
    DOFEffect,
    DOFMode,
    DOFQuality,
    DOFSettings,
    FarFieldDOF,
    NearFieldDOF,
)

# Motion Blur
from .motion_blur import (
    CameraMotion,
    CameraMotionBlur,
    MotionBlurEffect,
    MotionBlurMode,
    MotionBlurQuality,
    MotionBlurSettings,
    ObjectMotionBlur,
    TileMaxVelocity,
)

# Ambient Occlusion
from .ambient_occlusion import (
    AOEffect,
    AOMethod,
    AOQuality,
    AOSettings,
    BentNormalOutput,
    BilateralFilter,
    GTAO,
    HBAO,
    SSAO,
    SSAOKernel,
)

# Anti-Aliasing
from .antialiasing import (
    AAEffect,
    AAMethod,
    AASettings,
    FXAA,
    FXAAQuality,
    FXAASettings,
    JitterPattern,
    JitterSequence,
    SMAA,
    SMAAQuality,
    SMAASettings,
    TAA,
    TAASettings,
)

# Upscaling
from .upscaling import (
    BilinearUpscaler,
    CASUpscaler,
    DLSSUpscaler,
    FrameGenerationMode,
    FSR1Upscaler,
    FSR2Upscaler,
    get_render_resolution,
    SpatialUpscaler,
    TemporalUpscaler,
    UpscaleQuality,
    UpscaleResolution,
    UpscalerType,
    UpscalingEffect,
    UpscalingSettings,
    XeSSUpscaler,
)

# Constants
from .constants import (
    AA,
    AO,
    BLOOM,
    COLOR_GRADING,
    DOF,
    EPSILON,
    EXPOSURE,
    LUMINANCE_COEFFS_BT601,
    LUMINANCE_COEFFS_BT709,
    LUMINANCE_MIN,
    MOTION_BLUR,
    SAFE_LOG_MIN,
    TONEMAP,
    UPSCALING,
    calculate_luminance,
)


__all__ = [
    # Post-Process Stack
    "BlendMode",
    "EffectExecutionPath",
    "EffectPriority",
    "EffectQuality",
    "EffectSettings",
    "ExecutionFlags",
    "get_quality_preset",
    "IntermediateTarget",
    "IntermediateTargetManager",
    "PostProcessContext",
    "PostProcessEffect",
    "PostProcessStackConfig",
    "PostProcessStack",
    "PostProcessStackExecutor",
    "PostProcessVolume",
    "PostProcessVolumeSettings",
    "QUALITY_PRESETS",
    "QUALITY_PRESET_HIGH",
    "QUALITY_PRESET_LOW",
    "QUALITY_PRESET_MEDIUM",
    "QUALITY_PRESET_ULTRA",
    "QualityPreset",
    "SphereVolumeShape",
    "VolumeShape",
    "BoxVolumeShape",
    # Exposure
    "ExposureMode",
    "MeteringMode",
    "ExposureSettings",
    "luminance_to_ev",
    "ev_to_exposure",
    "exposure_to_ev",
    "ExposureCalculator",
    "ManualExposure",
    "AutoExposure",
    "HistogramExposure",
    "AdaptationCurve",
    "EyeAdaptation",
    "ExposureEffect",
    # Bloom
    "BlurMethod",
    "BloomQuality",
    "BloomMipSettings",
    "LensDirtSettings",
    "BloomSettings",
    "BloomThreshold",
    "BloomDownsample",
    "BloomBlur",
    "BloomUpsample",
    "BloomEffect",
    # Tone Mapping
    "TonemapOperator",
    "TonemapCurvePoint",
    "CustomCurveSettings",
    "TonemapSettings",
    "TonemapFunction",
    "Reinhard",
    "ReinhardExtended",
    "ACES",
    "ACESFitted",
    "AgX",
    "Filmic",
    "CustomCurve",
    "TonemappingEffect",
    # Color Grading
    "ColorSpace",
    "LUTFormat",
    "WhiteBalanceSettings",
    "LiftGammaGain",
    "ContrastSettings",
    "SaturationSettings",
    "HueSatLightness",
    "LUT3DSettings",
    "LUT3D",
    "ColorGradingSettings",
    "ColorGradingStack",
    "ColorGradingEffect",
    # Depth of Field
    "DOFMode",
    "BokehShapeType",
    "DOFQuality",
    "BokehShape",
    "CircleOfConfusion",
    "DOFSettings",
    "NearFieldDOF",
    "FarFieldDOF",
    "AutoFocusSystem",
    "DOFEffect",
    # Motion Blur
    "MotionBlurMode",
    "MotionBlurQuality",
    "MotionBlurSettings",
    "CameraMotion",
    "CameraMotionBlur",
    "ObjectMotionBlur",
    "TileMaxVelocity",
    "MotionBlurEffect",
    # Ambient Occlusion
    "AOMethod",
    "AOQuality",
    "AOSettings",
    "SSAOKernel",
    "SSAO",
    "HBAO",
    "GTAO",
    "BentNormalOutput",
    "BilateralFilter",
    "AOEffect",
    # Anti-Aliasing
    "AAMethod",
    "FXAAQuality",
    "SMAAQuality",
    "JitterPattern",
    "FXAASettings",
    "SMAASettings",
    "TAASettings",
    "AASettings",
    "JitterSequence",
    "FXAA",
    "SMAA",
    "TAA",
    "AAEffect",
    # Upscaling
    "UpscalerType",
    "UpscaleQuality",
    "FrameGenerationMode",
    "UpscaleResolution",
    "get_render_resolution",
    "UpscalingSettings",
    "SpatialUpscaler",
    "BilinearUpscaler",
    "FSR1Upscaler",
    "CASUpscaler",
    "TemporalUpscaler",
    "FSR2Upscaler",
    "DLSSUpscaler",
    "XeSSUpscaler",
    "UpscalingEffect",
    # Constants
    "EPSILON",
    "SAFE_LOG_MIN",
    "LUMINANCE_MIN",
    "EXPOSURE",
    "BLOOM",
    "TONEMAP",
    "DOF",
    "AO",
    "AA",
    "MOTION_BLUR",
    "UPSCALING",
    "COLOR_GRADING",
    "LUMINANCE_COEFFS_BT709",
    "LUMINANCE_COEFFS_BT601",
    "calculate_luminance",
]
