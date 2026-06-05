"""
Denoising System for Ray-Traced Signals

Provides comprehensive spatial and temporal denoising for ray-traced rendering:

Spatial Denoising (A-Trous Wavelet):
- ATrousDenoiser: Multi-pass wavelet filter with edge-stopping
- EdgeStopFunctions: Edge-preserving weights (depth, normal, luminance)
- DenoiseConfig: Configuration for spatial denoising
- WaveletKernel: 5x5 A-Trous wavelet filter kernel

Temporal Denoising:
- TemporalDenoiser: Variance-guided temporal accumulation
- VarianceGuided: Adaptive blend weights based on local variance
- HistoryTracker: Per-pixel history length tracking (1-64 frames)
- NeighbourhoodClamper: Ghost rejection via AABB/variance clamping
- Reprojector: Velocity buffer reprojection with disocclusion detection
- EMABlender: Exponential moving average temporal blend

Common:
- YCoCgConverter: RGB to YCoCg color space conversion
- DenoiseTarget/TemporalTarget: Signal type selection (GI, reflections, shadows)

WGSL Shaders (referenced):
- denoise_atrous.comp.wgsl: A-Trous spatial filter
- denoise_temporal.comp.wgsl: Temporal accumulation pass
- denoise_edge_stop.wgsl: Shared edge-stopping functions
"""

from .atrous_denoiser import (
    # Core Denoiser
    ATrousDenoiser,
    ATrousPass,
    # Configuration
    DenoiseConfig,
    DenoiseQuality,
    DenoiseTarget,
    # Edge-Stopping Functions
    EdgeStopFunctions,
    EdgeStopWeights,
    DepthEdgeStop,
    NormalEdgeStop,
    LuminanceEdgeStop,
    # Color Space Conversion
    YCoCgConverter,
    # Filter Kernel
    WaveletKernel,
    GAUSSIAN_5X5_KERNEL,
    # Buffers
    PingPongBuffers,
    DenoiseGBuffer,
    # Quality Metrics
    PSNRMetrics,
    DenoiseStats,
    # Convenience Functions
    create_gi_denoiser,
    create_reflection_denoiser,
    create_shadow_denoiser,
    create_default_config,
    create_quality_config,
)

from .temporal_denoiser import (
    # Core Denoiser
    TemporalDenoiser,
    # Configuration
    TemporalDenoiseConfig,
    TemporalQuality,
    TemporalTarget,
    ClampingMode,
    DisocclusionMode,
    QualityPreset,
    QUALITY_PRESETS,
    # Components
    VarianceGuided,
    VarianceEstimate,
    HistoryTracker,
    HistoryEntry,
    NeighbourhoodClamper,
    Reprojector,
    ReprojectionResult,
    EMABlender,
    # Buffers
    TemporalBuffer,
    TemporalBufferSet,
    TemporalGBuffer,
    # Statistics
    TemporalDenoiseStats,
    # Constants
    MIN_HISTORY_LENGTH,
    MAX_HISTORY_LENGTH,
    DEFAULT_CONVERGENCE_FRAMES,
    DEFAULT_VARIANCE_GAMMA,
    DEFAULT_EMA_ALPHA,
    EPSILON,
    # Factory Functions
    create_gi_temporal_denoiser,
    create_reflection_temporal_denoiser,
    create_shadow_temporal_denoiser,
    create_fast_temporal_denoiser,
    create_quality_temporal_denoiser,
)

from .svgf_denoiser import (
    # Core SVGF Denoiser
    SVGFDenoiser,
    SVGFConfig,
    SVGFStats,
    SVGFQuality,
    FilterMode,
    # Variance Estimation
    VarianceEstimator,
    VarianceEstimate as SVGFVarianceEstimate,
    # Temporal Accumulation
    TemporalAccumulator,
    TemporalAccumulationState,
    TemporalSample,
    TemporalBufferSet as SVGFTemporalBufferSet,
    # Disocclusion Detection
    DisocclusionDetector,
    DisocclusionMode as SVGFDisocclusionMode,
    DisocclusionResult,
    # Spatiotemporal Filter
    SpatiotemporalFilter,
    SpatiotemporalFilterConfig,
    # Comparison
    DenoiserComparison,
    compare_denoisers,
    # Convenience Functions
    create_svgf_denoiser,
    create_gi_svgf_denoiser,
    create_reflection_svgf_denoiser,
    create_pathtracing_svgf_denoiser,
    # Constants
    VARIANCE_NEIGHBOURHOOD_SIZE,
    VARIANCE_MIN_SAMPLES,
    VARIANCE_CLAMP_MAX,
    VARIANCE_GAMMA,
    TEMPORAL_MIN_ALPHA,
    TEMPORAL_MAX_ALPHA,
    TEMPORAL_CONVERGE_FRAMES,
    DEPTH_REJECT_THRESHOLD,
    NORMAL_REJECT_THRESHOLD,
    VELOCITY_REJECT_THRESHOLD,
    FIREFLY_THRESHOLD,
)

__all__ = [
    # === Spatial Denoising (A-Trous) ===
    # Core Denoiser
    "ATrousDenoiser",
    "ATrousPass",
    # Configuration
    "DenoiseConfig",
    "DenoiseQuality",
    "DenoiseTarget",
    # Edge-Stopping Functions
    "EdgeStopFunctions",
    "EdgeStopWeights",
    "DepthEdgeStop",
    "NormalEdgeStop",
    "LuminanceEdgeStop",
    # Color Space Conversion
    "YCoCgConverter",
    # Filter Kernel
    "WaveletKernel",
    "GAUSSIAN_5X5_KERNEL",
    # Buffers
    "PingPongBuffers",
    "DenoiseGBuffer",
    # Quality Metrics
    "PSNRMetrics",
    "DenoiseStats",
    # Convenience Functions
    "create_gi_denoiser",
    "create_reflection_denoiser",
    "create_shadow_denoiser",
    "create_default_config",
    "create_quality_config",
    # === Temporal Denoising ===
    # Core Denoiser
    "TemporalDenoiser",
    # Configuration
    "TemporalDenoiseConfig",
    "TemporalQuality",
    "TemporalTarget",
    "ClampingMode",
    "DisocclusionMode",
    "QualityPreset",
    "QUALITY_PRESETS",
    # Components
    "VarianceGuided",
    "VarianceEstimate",
    "HistoryTracker",
    "HistoryEntry",
    "NeighbourhoodClamper",
    "Reprojector",
    "ReprojectionResult",
    "EMABlender",
    # Buffers
    "TemporalBuffer",
    "TemporalBufferSet",
    "TemporalGBuffer",
    # Statistics
    "TemporalDenoiseStats",
    # Constants
    "MIN_HISTORY_LENGTH",
    "MAX_HISTORY_LENGTH",
    "DEFAULT_CONVERGENCE_FRAMES",
    "DEFAULT_VARIANCE_GAMMA",
    "DEFAULT_EMA_ALPHA",
    "EPSILON",
    # Factory Functions
    "create_gi_temporal_denoiser",
    "create_reflection_temporal_denoiser",
    "create_shadow_temporal_denoiser",
    "create_fast_temporal_denoiser",
    "create_quality_temporal_denoiser",
    # === SVGF (Spatiotemporal Variance-Guided Filtering) ===
    # Core SVGF Denoiser
    "SVGFDenoiser",
    "SVGFConfig",
    "SVGFStats",
    "SVGFQuality",
    "FilterMode",
    # Variance Estimation
    "VarianceEstimator",
    "SVGFVarianceEstimate",
    # Temporal Accumulation
    "TemporalAccumulator",
    "TemporalAccumulationState",
    "TemporalSample",
    "SVGFTemporalBufferSet",
    # Disocclusion Detection
    "DisocclusionDetector",
    "SVGFDisocclusionMode",
    "DisocclusionResult",
    # Spatiotemporal Filter
    "SpatiotemporalFilter",
    "SpatiotemporalFilterConfig",
    # Comparison
    "DenoiserComparison",
    "compare_denoisers",
    # Convenience Functions
    "create_svgf_denoiser",
    "create_gi_svgf_denoiser",
    "create_reflection_svgf_denoiser",
    "create_pathtracing_svgf_denoiser",
    # SVGF Constants
    "VARIANCE_NEIGHBOURHOOD_SIZE",
    "VARIANCE_MIN_SAMPLES",
    "VARIANCE_CLAMP_MAX",
    "VARIANCE_GAMMA",
    "TEMPORAL_MIN_ALPHA",
    "TEMPORAL_MAX_ALPHA",
    "TEMPORAL_CONVERGE_FRAMES",
    "DEPTH_REJECT_THRESHOLD",
    "NORMAL_REJECT_THRESHOLD",
    "VELOCITY_REJECT_THRESHOLD",
    "FIREFLY_THRESHOLD",
]
