"""Lighting and Shadows subsystem for the rendering engine.

This module provides comprehensive lighting support including:
- Direct lights (directional, point, spot, area, IES, sky)
- Shadow mapping (CSM, cube, spot shadows)
- Shadow filtering (PCF, PCSS, VSM, ESM, contact shadows)
- Clustered light culling (3D froxels)
- Global illumination (light probes, DDGI, reflection probes)

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
    # Quality presets and camera-relative grid (T-GIR-P2.1)
    DDGIQualityPreset,
    DDGIConfig,
    DDGICameraRelativeGrid,
    get_preset_params,
    estimate_gpu_memory,
    _QUALITY_PARAMS,
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
    # Quality presets and camera-relative grid (T-GIR-P2.1)
    "DDGIQualityPreset",
    "DDGIConfig",
    "DDGICameraRelativeGrid",
    "get_preset_params",
    "estimate_gpu_memory",
    "_QUALITY_PARAMS",
]
