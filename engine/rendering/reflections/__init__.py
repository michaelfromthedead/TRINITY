"""
Reflection Systems for TRINITY Engine

Provides screen-space, temporal, and ray-traced reflection implementations:
- SSRRoughnessBlur: Roughness-driven blur for glossy SSR
- GaussianBlur: Separable Gaussian convolution
- BilateralUpscale: Edge-aware upsampling
- MaterialReflectionParams: Per-material reflection configuration
- SSRTemporalReprojection: Temporal reprojection for stable SSR
- TemporalBuffer: Ping-pong history buffers with confidence tracking
- RTReflectionPass: Hardware ray-traced reflections (T-GIR-P8.1)
- GBufferReader: G-Buffer data extraction
- ReflectionRayGenerator: Reflection ray computation
- RTReflectionTracer: TLAS ray tracing interface
- RoughnessRayMapping: Roughness-based ray count adaptation (T-GIR-P8.3)
- ResolutionHierarchy: Multi-resolution rendering tiers
- AdaptiveRayScheduler: Temporal stratification for ray scheduling
- RayBudgetManager: Global ray budget distribution
- AdaptiveRTPass: Complete adaptive ray tracing pass
- ReflectionFallbackPass: Per-pixel technique selection (T-GIR-P8.5)
- TechniqueSelector: RT -> SSR -> Probe -> Environment fallback chain
- ConfidenceBlender: Confidence-based blending between techniques
- TransitionManager: Temporal smoothing to prevent popping
"""

from .ssr_blur import (
    BlurTechnique,
    SSRBlurQuality,
    SSRBlurConstants,
    SSR_BLUR,
    MaterialReflectionParams,
    GaussianBlur,
    BilateralUpscale,
    DownsampleChain,
    SSRRoughnessBlur,
    SSRRoughnessBlurSettings,
)

from .ssr_temporal import (
    DisocclusionMode,
    SSRTemporalConfig,
    SSRTemporalReprojection,
    TemporalBuffer,
    TemporalBufferSet,
    TemporalQuality,
    TemporalSample,
    TemporalStats,
)

from .rt_reflections import (
    # Constants
    DEFAULT_ROUGHNESS_THRESHOLD,
    DEFAULT_MAX_RAY_DISTANCE,
    DEFAULT_NORMAL_BIAS,
    DEFAULT_ENVIRONMENT_COLOR,
    RAY_FLAG_NONE,
    RAY_FLAG_CULL_BACK_FACING,
    RAY_FLAG_CULL_FRONT_FACING,
    RAY_FLAG_TERMINATE_ON_FIRST_HIT,
    RAY_FLAG_SKIP_CLOSEST_HIT,
    RAY_FLAG_ACCEPT_FIRST_HIT,
    RESOLUTION_QUARTER,
    RESOLUTION_HALF,
    RESOLUTION_FULL,
    # Enums
    ResolutionMode,
    # Data structures
    MaterialData,
    GBufferPixel,
    ReflectionRay,
    RayHitInfo,
    ReflectionOutput,
    # Config
    RTReflectionConfig,
    # Core classes
    GBufferReader,
    ReflectionRayGenerator,
    RTReflectionTracer,
    RTReflectionPass,
    # TLAS
    TLASInterface,
    MockTLAS,
    # Utilities
    generate_rt_reflections_rgen_wgsl,
    estimate_rt_reflection_memory,
    create_mock_tlas,
)

from .rt_brdf_sampling import (
    # Constants
    PI as BRDF_PI,
    TWO_PI as BRDF_TWO_PI,
    INV_PI as BRDF_INV_PI,
    EPSILON as BRDF_EPSILON,
    MIN_ROUGHNESS,
    MAX_ROUGHNESS,
    # GGX Distribution
    GGXMicrofacetDistribution,
    # Tangent Basis
    TangentBasis,
    # Importance Sampling
    ImportanceSamplerGGX,
    # BRDF Evaluation
    BRDFEvaluator,
    # Sample Result
    SampleResult,
    # Main Sampler
    RTBRDFSampler,
    # Shader Generation
    generate_rt_brdf_rchit_wgsl,
    # Utilities
    compute_f0_from_ior,
    compute_f0_for_metal,
    lerp_f0,
)

from .rt_adaptive_rays import (
    # Constants
    DEFAULT_TIER_THRESHOLDS,
    DEFAULT_RESOLUTION_SCALES,
    DEFAULT_RAY_COUNTS,
    DEFAULT_DENOISE_STRENGTHS,
    DEFAULT_RAY_BUDGET,
    DEFAULT_TEMPORAL_FRAMES,
    MAX_RT_ROUGHNESS,
    # Enums
    ResolutionTier,
    DenoiseLevel,
    # Data structures
    RoughnessMapping,
    ResolutionLevel,
    ScheduledRays,
    AccumulatedResult,
    BudgetAllocation,
    AdaptiveRayResult,
    # Core classes
    RoughnessRayMapping,
    ResolutionHierarchy,
    AdaptiveRayScheduler,
    RayBudgetManager,
    AdaptiveRTPass,
    # Config
    AdaptiveRTConfig,
    # Utilities
    generate_adaptive_rays_wgsl,
    estimate_adaptive_memory,
    create_test_roughness_map,
)

from .reflection_fallback import (
    # Constants
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_BLEND_THRESHOLD,
    DEFAULT_TRANSITION_SPEED,
    MIN_VALID_CONFIDENCE,
    DEFAULT_HISTORY_LENGTH,
    # Enums
    ReflectionTechnique,
    # Data structures
    TechniqueResult,
    FallbackPassOutput,
    PixelHistory,
    # Configuration
    FallbackChainConfig,
    # Core classes
    TechniqueSelector,
    ConfidenceBlender,
    TransitionManager,
    ReflectionFallbackPass,
    # Utilities
    generate_fallback_chain_wgsl,
    evaluate_fallback_chain,
)

from .rt_reflection_denoise import (
    # Constants
    EPSILON as DENOISE_EPSILON,
    LUMINANCE_EPSILON,
    DEPTH_EPSILON,
    DEFAULT_SIGMA_DEPTH,
    DEFAULT_SIGMA_NORMAL,
    DEFAULT_SIGMA_LUMINANCE,
    DEFAULT_TEMPORAL_ALPHA,
    DEFAULT_HISTORY_FRAMES,
    DEFAULT_ATROUS_ITERATIONS,
    MAX_ATROUS_ITERATIONS,
    DEFAULT_DILATIONS,
    GAUSSIAN_5X5_KERNEL,
    BILATERAL_RADIUS,
    # Quality
    ReflectionDenoiseQuality,
    QualityPresetParams,
    QUALITY_PRESETS as DENOISE_QUALITY_PRESETS,
    # Color Space
    YCoCgConverter as DenoiseYCoCgConverter,
    # Edge-Stopping
    EdgeStopWeights,
    ReflectionEdgeStopFunctions,
    # A-Trous Filter
    ATrousIterationResult,
    ATrousFilterResult,
    ReflectionATrousFilter,
    # Temporal Accumulation
    ReprojectionResult,
    TemporalAccumulationResult,
    ReflectionTemporalAccumulator,
    # Bilateral Upscale
    BilateralUpscaleResult,
    ReflectionBilateralUpscale,
    # Configuration
    RTReflectionDenoiseConfig,
    # Pipeline
    DenoisePipelineResult,
    RTReflectionDenoisePipeline,
    # WGSL Generation
    generate_rt_reflections_denoise_wgsl,
    # Factory Functions
    create_reflection_denoiser,
    create_fast_reflection_denoiser,
    create_quality_reflection_denoiser,
)

__all__ = [
    # SSR Blur
    "BlurTechnique",
    "SSRBlurQuality",
    "SSRBlurConstants",
    "SSR_BLUR",
    "MaterialReflectionParams",
    "GaussianBlur",
    "BilateralUpscale",
    "DownsampleChain",
    "SSRRoughnessBlur",
    "SSRRoughnessBlurSettings",
    # SSR Temporal
    "DisocclusionMode",
    "SSRTemporalConfig",
    "SSRTemporalReprojection",
    "TemporalBuffer",
    "TemporalBufferSet",
    "TemporalQuality",
    "TemporalSample",
    "TemporalStats",
    # RT Reflections (T-GIR-P8.1)
    "DEFAULT_ROUGHNESS_THRESHOLD",
    "DEFAULT_MAX_RAY_DISTANCE",
    "DEFAULT_NORMAL_BIAS",
    "DEFAULT_ENVIRONMENT_COLOR",
    "RAY_FLAG_NONE",
    "RAY_FLAG_CULL_BACK_FACING",
    "RAY_FLAG_CULL_FRONT_FACING",
    "RAY_FLAG_TERMINATE_ON_FIRST_HIT",
    "RAY_FLAG_SKIP_CLOSEST_HIT",
    "RAY_FLAG_ACCEPT_FIRST_HIT",
    "RESOLUTION_QUARTER",
    "RESOLUTION_HALF",
    "RESOLUTION_FULL",
    "ResolutionMode",
    "MaterialData",
    "GBufferPixel",
    "ReflectionRay",
    "RayHitInfo",
    "ReflectionOutput",
    "RTReflectionConfig",
    "GBufferReader",
    "ReflectionRayGenerator",
    "RTReflectionTracer",
    "RTReflectionPass",
    "TLASInterface",
    "MockTLAS",
    "generate_rt_reflections_rgen_wgsl",
    "estimate_rt_reflection_memory",
    "create_mock_tlas",
    # RT Adaptive Rays (T-GIR-P8.3)
    "DEFAULT_TIER_THRESHOLDS",
    "DEFAULT_RESOLUTION_SCALES",
    "DEFAULT_RAY_COUNTS",
    "DEFAULT_DENOISE_STRENGTHS",
    "DEFAULT_RAY_BUDGET",
    "DEFAULT_TEMPORAL_FRAMES",
    "MAX_RT_ROUGHNESS",
    "ResolutionTier",
    "DenoiseLevel",
    "RoughnessMapping",
    "ResolutionLevel",
    "ScheduledRays",
    "AccumulatedResult",
    "BudgetAllocation",
    "AdaptiveRayResult",
    "RoughnessRayMapping",
    "ResolutionHierarchy",
    "AdaptiveRayScheduler",
    "RayBudgetManager",
    "AdaptiveRTPass",
    "AdaptiveRTConfig",
    "generate_adaptive_rays_wgsl",
    "estimate_adaptive_memory",
    "create_test_roughness_map",
    # RT BRDF Sampling (T-GIR-P8.2)
    "BRDF_PI",
    "BRDF_TWO_PI",
    "BRDF_INV_PI",
    "BRDF_EPSILON",
    "MIN_ROUGHNESS",
    "MAX_ROUGHNESS",
    "GGXMicrofacetDistribution",
    "TangentBasis",
    "ImportanceSamplerGGX",
    "BRDFEvaluator",
    "SampleResult",
    "RTBRDFSampler",
    "generate_rt_brdf_rchit_wgsl",
    "compute_f0_from_ior",
    "compute_f0_for_metal",
    "lerp_f0",
    # RT Reflection Denoise (T-GIR-P8.4)
    "DENOISE_EPSILON",
    "LUMINANCE_EPSILON",
    "DEPTH_EPSILON",
    "DEFAULT_SIGMA_DEPTH",
    "DEFAULT_SIGMA_NORMAL",
    "DEFAULT_SIGMA_LUMINANCE",
    "DEFAULT_TEMPORAL_ALPHA",
    "DEFAULT_HISTORY_FRAMES",
    "DEFAULT_ATROUS_ITERATIONS",
    "MAX_ATROUS_ITERATIONS",
    "DEFAULT_DILATIONS",
    "GAUSSIAN_5X5_KERNEL",
    "BILATERAL_RADIUS",
    "ReflectionDenoiseQuality",
    "QualityPresetParams",
    "DENOISE_QUALITY_PRESETS",
    "DenoiseYCoCgConverter",
    "EdgeStopWeights",
    "ReflectionEdgeStopFunctions",
    "ATrousIterationResult",
    "ATrousFilterResult",
    "ReflectionATrousFilter",
    "ReprojectionResult",
    "TemporalAccumulationResult",
    "ReflectionTemporalAccumulator",
    "BilateralUpscaleResult",
    "ReflectionBilateralUpscale",
    "RTReflectionDenoiseConfig",
    "DenoisePipelineResult",
    "RTReflectionDenoisePipeline",
    "generate_rt_reflections_denoise_wgsl",
    "create_reflection_denoiser",
    "create_fast_reflection_denoiser",
    "create_quality_reflection_denoiser",
    # RT Reflection Fallback Chain (T-GIR-P8.5)
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_BLEND_THRESHOLD",
    "DEFAULT_TRANSITION_SPEED",
    "MIN_VALID_CONFIDENCE",
    "DEFAULT_HISTORY_LENGTH",
    "ReflectionTechnique",
    "TechniqueResult",
    "FallbackPassOutput",
    "PixelHistory",
    "FallbackChainConfig",
    "TechniqueSelector",
    "ConfidenceBlender",
    "TransitionManager",
    "ReflectionFallbackPass",
    "generate_fallback_chain_wgsl",
    "evaluate_fallback_chain",
]
