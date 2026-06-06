"""Lighting and Shadows subsystem for the rendering engine.

This module provides comprehensive lighting support including:
- Direct lights (directional, point, spot, area, IES, sky)
- Shadow mapping (CSM, cube, spot shadows)
- Shadow filtering (PCF, PCSS, VSM, ESM, contact shadows)
- Clustered light culling (3D froxels)
- Global illumination (light probes, DDGI, reflection probes)
- Planar reflections (mirrors, water surfaces)

See RENDERING_CONTEXT.md Section 6.4 for full specification.
"""

# Constants
from engine.rendering.lighting.constants import (
    ShadowConstants,
    CSMConstants,
    FroxelConstants,
    LightConstants,
    GIProbeConstants,
    DDGIConstants,
    ShadowFilterConstants,
    EPSILON,
)

# Light types
from engine.rendering.lighting.light_types import (
    # Enums
    LightType,
    ShadowMode,
    GIImportance,
    # Configs
    ShadowCasterConfig,
    GIContributorConfig,
    # Decorators
    shadow_caster,
    gi_contributor,
    # Base class
    Light,
    # Light types
    DirectionalLight,
    PointLight,
    SpotLight,
    RectAreaLight,
    DiskAreaLight,
    IESLight,
    IESProfile,
    SkyLight,
    # Type alias
    AnyLight,
)

# Light culling
from engine.rendering.lighting.light_culling import (
    FroxelBounds,
    Froxel,
    FroxelGridConfig,
    FroxelGrid,
    LightList,
    ClusteredLightCuller,
)

# Shadow mapping
from engine.rendering.lighting.shadows import (
    ShadowMapType,
    ShadowMapConfig,
    ShadowMap,
    CascadeData,
    CascadedShadowMap,
    CubeShadowMap,
    SpotShadowMap,
    ShadowAtlasSlot,
    ShadowAtlas,
)

# Shadow filtering
from engine.rendering.lighting.shadow_filtering import (
    ShadowFilterType,
    ShadowSample,
    ShadowFilter,
    ShadowMapSampler,
    PCFConfig,
    PCFFilter,
    PCSSConfig,
    PCSSFilter,
    VSMConfig,
    VSMFilter,
    ESMConfig,
    ESMFilter,
    ContactShadowConfig,
    ContactShadowFilter,
    create_shadow_filter,
)

# GI probes
from engine.rendering.lighting.gi_probes import (
    ProbeType,
    CaptureMode,
    SphericalHarmonics,
    LightProbe,
    ProbeGridConfig,
    ProbeGrid,
    IrradianceVolume,
    LightmapTexel,
    BakedLightmap,
    ReflectionProbeConfig,
    ReflectionProbe,
    reflection_probe,
)

# DDGI
from engine.rendering.lighting.gi_ddgi import (
    DDGIProbeState,
    DDGIProbeConfig,
    DDGIProbe,
    DDGIGridConfig,
    DDGIProbeGrid,
    RayResult,
    DDGIUpdatePass,
    DDGILookup,
)

# Planar mirrors
from engine.rendering.lighting.planar_mirror import (
    MirrorUpdateMode,
    PlanarMirrorConfig,
    PlanarMirror,
    PlanarMirrorManager,
    create_water_plane,
    create_mirror_plane,
)

# Baked probes
from engine.rendering.lighting.baked_probes import (
    BakedProbeConstants,
    CubemapFace,
    CUBEMAP_FACE_DIRECTIONS,
    CompressionQuality,
    FilterMode,
    HDRPixel,
    CubemapFaceData,
    CubemapData,
    MipLevel,
    CubemapMipChain,
    BC6HBlock,
    BC6HCompressor,
    KTX2Format,
    KTX2Header,
    KTX2Writer,
    KTX2Reader,
    CaptureConfig,
    CubemapRenderer,
    FunctionCubemapRenderer,
    MipGenerator,
    PrefilteredGenerator,
    BakedProbeConfig,
    BakedProbeAsset,
    BakedProbeCapture,
    BakedProbeManager,
)

# Realtime reflection probes
from engine.rendering.lighting.reflection_probes import (
    RealtimeProbeConstants,
    SchedulerMode,
    ProbeUpdateReason,
    FaceState,
    RealtimeProbeFaceScheduler,
    RealtimeProbeCaptureSettings,
    DynamicObjectFilter,
    RealtimeProbeCapture,
    FunctionRealtimeProbeCapture,
    RealtimeProbeState,
    RealtimeReflectionProbe,
    CaptureBudget,
    RealtimeProbeManager,
    HybridProbeMode,
    HybridProbeConfig,
    HybridReflectionProbe,
)

# Probe atlas management
from engine.rendering.lighting.probe_atlas import (
    AtlasConstants,
    AtlasFormat,
    AtlasSlot,
    AtlasLayout,
    AtlasConfig,
    ProbeAtlas,
    PendingUpdate,
    AtlasUpdater,
    AtlasSampler,
    ProbeAtlasManager,
)

# Probe blending
from engine.rendering.lighting.probe_blending import (
    ProbeBlendConstants,
    FalloffType,
    ProbeInfluence,
    ProbeCollectorConfig,
    ProbeCollector,
    BlendResult,
    ProbeBlender,
    ProbeBlendConfig,
    GBufferSample,
    ReflectionBuffer,
    ProbeBlendPass,
    generate_probe_blend_wgsl,
    generate_probe_blend_shader,
)

# Parallax correction
from engine.rendering.lighting.probe_parallax import (
    ParallaxConstants,
    BoxFace,
    ProbeBox,
    RayBoxIntersection,
    ParallaxConfig,
    ParallaxCorrector,
    ParallaxProbeAdapter,
    compute_box_projection_direction,
    blend_directions,
)

# Pre-filtered cubemaps
from engine.rendering.lighting.probe_prefilter import (
    PrefilterConstants,
    GGXDistribution,
    radical_inverse_vdc,
    hammersley,
    tangent_to_world,
    ImportanceSampler,
    PrefilterConfig,
    CubemapPrefilter,
    BRDFTerms,
    SplitSumLUT,
    PrefilterResult,
    PrefilterPipeline,
)

__all__ = [
    # Constants
    "ShadowConstants",
    "CSMConstants",
    "FroxelConstants",
    "LightConstants",
    "GIProbeConstants",
    "DDGIConstants",
    "ShadowFilterConstants",
    "EPSILON",
    # Light types
    "LightType",
    "ShadowMode",
    "GIImportance",
    "ShadowCasterConfig",
    "GIContributorConfig",
    "shadow_caster",
    "gi_contributor",
    "Light",
    "DirectionalLight",
    "PointLight",
    "SpotLight",
    "RectAreaLight",
    "DiskAreaLight",
    "IESLight",
    "IESProfile",
    "SkyLight",
    "AnyLight",
    # Light culling
    "FroxelBounds",
    "Froxel",
    "FroxelGridConfig",
    "FroxelGrid",
    "LightList",
    "ClusteredLightCuller",
    # Shadow mapping
    "ShadowMapType",
    "ShadowMapConfig",
    "ShadowMap",
    "CascadeData",
    "CascadedShadowMap",
    "CubeShadowMap",
    "SpotShadowMap",
    "ShadowAtlasSlot",
    "ShadowAtlas",
    # Shadow filtering
    "ShadowFilterType",
    "ShadowSample",
    "ShadowFilter",
    "ShadowMapSampler",
    "PCFConfig",
    "PCFFilter",
    "PCSSConfig",
    "PCSSFilter",
    "VSMConfig",
    "VSMFilter",
    "ESMConfig",
    "ESMFilter",
    "ContactShadowConfig",
    "ContactShadowFilter",
    "create_shadow_filter",
    # GI probes
    "ProbeType",
    "CaptureMode",
    "SphericalHarmonics",
    "LightProbe",
    "ProbeGridConfig",
    "ProbeGrid",
    "IrradianceVolume",
    "LightmapTexel",
    "BakedLightmap",
    "ReflectionProbeConfig",
    "ReflectionProbe",
    "reflection_probe",
    # DDGI
    "DDGIProbeState",
    "DDGIProbeConfig",
    "DDGIProbe",
    "DDGIGridConfig",
    "DDGIProbeGrid",
    "RayResult",
    "DDGIUpdatePass",
    "DDGILookup",
    # Planar mirrors
    "MirrorUpdateMode",
    "PlanarMirrorConfig",
    "PlanarMirror",
    "PlanarMirrorManager",
    "create_water_plane",
    "create_mirror_plane",
    # Baked probes
    "BakedProbeConstants",
    "CubemapFace",
    "CUBEMAP_FACE_DIRECTIONS",
    "CompressionQuality",
    "FilterMode",
    "HDRPixel",
    "CubemapFaceData",
    "CubemapData",
    "MipLevel",
    "CubemapMipChain",
    "BC6HBlock",
    "BC6HCompressor",
    "KTX2Format",
    "KTX2Header",
    "KTX2Writer",
    "KTX2Reader",
    "CaptureConfig",
    "CubemapRenderer",
    "FunctionCubemapRenderer",
    "MipGenerator",
    "PrefilteredGenerator",
    "BakedProbeConfig",
    "BakedProbeAsset",
    "BakedProbeCapture",
    "BakedProbeManager",
    # Realtime reflection probes
    "RealtimeProbeConstants",
    "SchedulerMode",
    "ProbeUpdateReason",
    "FaceState",
    "RealtimeProbeFaceScheduler",
    "RealtimeProbeCaptureSettings",
    "DynamicObjectFilter",
    "RealtimeProbeCapture",
    "FunctionRealtimeProbeCapture",
    "RealtimeProbeState",
    "RealtimeReflectionProbe",
    "CaptureBudget",
    "RealtimeProbeManager",
    "HybridProbeMode",
    "HybridProbeConfig",
    "HybridReflectionProbe",
    # Probe atlas management
    "AtlasConstants",
    "AtlasFormat",
    "AtlasSlot",
    "AtlasLayout",
    "AtlasConfig",
    "ProbeAtlas",
    "PendingUpdate",
    "AtlasUpdater",
    "AtlasSampler",
    "ProbeAtlasManager",
    # Probe blending
    "ProbeBlendConstants",
    "FalloffType",
    "ProbeInfluence",
    "ProbeCollectorConfig",
    "ProbeCollector",
    "BlendResult",
    "ProbeBlender",
    "ProbeBlendConfig",
    "GBufferSample",
    "ReflectionBuffer",
    "ProbeBlendPass",
    "generate_probe_blend_wgsl",
    "generate_probe_blend_shader",
    # Parallax correction
    "ParallaxConstants",
    "BoxFace",
    "ProbeBox",
    "RayBoxIntersection",
    "ParallaxConfig",
    "ParallaxCorrector",
    "ParallaxProbeAdapter",
    "compute_box_projection_direction",
    "blend_directions",
    # Pre-filtered cubemaps
    "PrefilterConstants",
    "GGXDistribution",
    "radical_inverse_vdc",
    "hammersley",
    "tangent_to_world",
    "ImportanceSampler",
    "PrefilterConfig",
    "CubemapPrefilter",
    "BRDFTerms",
    "SplitSumLUT",
    "PrefilterResult",
    "PrefilterPipeline",
]
