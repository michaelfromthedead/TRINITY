"""TRINITY Material System: Pythonic surface shader authoring.

This package provides the material DSL for writing GPU shaders in Python
that compile to WGSL.

Core Classes:
    Material: Base class for user-defined materials
    MaterialMeta: Metaclass that compiles surface() to WGSL
    SurfaceContext: Input proxy for shader inputs (textures, UVs, etc.)
    SurfaceOutput: Output proxy for PBR material parameters
    MaterialCompiler: Compiles materials to WGSL/SPIR-V

Texture Types:
    Texture2D: 2D texture descriptor with WGSL binding generation
    TextureCube: Cubemap texture descriptor for environment maps

Vector Types:
    Vec2, Vec3, Vec4: WGSL vector proxies for material authoring

Decorators:
    @surface: Marks the shader entry point method

Example::

    from trinity.materials import Material, MaterialMeta, surface
    from trinity.materials import SurfaceContext, SurfaceOutput, Vec3
    from trinity.materials import Texture2D

    class BrickMaterial(Material, metaclass=MaterialMeta):
        albedo = Texture2D(default="white", srgb=True)
        normal = Texture2D(default="flat_normal")

        @surface
        def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
            out.base_color = ctx.sample(self.albedo, ctx.uv).xyz
            out.normal = ctx.sample(self.normal, ctx.uv).xyz * 2.0 - 1.0
            out.roughness = 0.8

    # Access compiled WGSL:
    print(BrickMaterial._wgsl_source)
"""

from trinity.materials.dsl import (
    Material,
    MaterialMeta,
    SurfaceContext,
    SurfaceOutput,
    surface,
    Vec2,
    Vec3,
    Vec4,
    PythonToWGSLTranslator,
    WGSLTranslationError,
)

from trinity.materials.compiler import MaterialCompiler

from trinity.materials.textures import (
    Texture2D,
    TextureCube,
    TextureDescriptor,
    TextureBindingSet,
    FilterMode,
    AddressMode,
    TextureFormat,
    DEFAULT_TEXTURES,
    get_default_texture,
    is_valid_default,
    collect_texture_descriptors,
)

from trinity.materials.builtins import (
    # Registry functions
    BUILTIN_REGISTRY,
    get_builtin_wgsl,
    get_required_builtins,
    # Noise functions
    value_noise,
    perlin_noise,
    simplex_noise,
    worley_noise,
    fbm,
    # Color conversion
    rgb_to_hsv,
    hsv_to_rgb,
    linear_to_srgb,
    srgb_to_linear,
    tonemap_reinhard,
    tonemap_aces,
)

from trinity.materials.variants import (
    MaterialDomain as VariantDomain,
    BlendMode as VariantBlendMode,
    QualityTier,
    VariantConfig,
    VariantCompiler,
    generate_all_variant_combinations,
    get_variant_for_material_system,
)

from trinity.materials.quality import (
    QualityFeatures,
    QualityShaderCode,
    get_quality_config_for_device,
)

from trinity.materials.blends import (
    BlendFactor,
    BlendOperation,
    ColorWriteMask,
    BlendState,
    BlendShaderCode,
    get_blend_state_for_variant,
    validate_blend_combination,
)

from trinity.materials.includes import (
    IncludeResolver,
    IncludeError,
    CyclicIncludeError,
    MaxDepthError,
    IncludeFileNotFoundError,
    IncludeDirective,
    DepGraph,
    preprocess_wgsl,
)

from trinity.materials.domains import (
    DomainCapability,
    DomainOutputFormat,
    DomainShaderTemplate,
    DomainVariantGenerator,
    DOMAIN_CAPABILITIES,
    DOMAIN_OUTPUT_FORMATS,
    domain_has_capability,
    get_domain_shader_info,
)

from trinity.materials.ui_domain import (
    UIBlendMode,
    UIMaterialConfig,
    UIMaterialBuilder,
    UI_DOMAIN_WGSL,
    UI_DOMAIN_MINIMAL_WGSL,
    UI_MATERIAL_PRESETS,
    generate_ui_material,
    generate_ui_material_consts,
    get_ui_entry_point,
    get_ui_material_preset,
    validate_ui_material_wgsl,
)

from trinity.materials.dep_graph import MaterialDepGraph

from trinity.materials.hot_reload import (
    HotReloadConfig,
    HotReloadWatcher,
    HotReloadManager,
    CompilationResult,
    MaterialFileHandler,
    WATCHDOG_AVAILABLE,
)

from trinity.materials.pbr_types import (
    PBRInput,
    PBRParams,
    PBROutput,
    get_pbr_structs_wgsl,
    PBR_STRUCTS_WGSL,
    PBR_PARAMS_FIELDS,
)

from trinity.materials.brdf import (
    # WGSL source access
    get_brdf_wgsl,
    # NDF functions
    d_ggx,
    # Geometry functions
    g1_schlick_ggx,
    g_smith_ggx,
    g_smith_schlick,
    # Fresnel functions
    f_schlick,
    f_schlick_roughness,
    f_schlick_scalar,
    # Diffuse BRDF
    brdf_diffuse,
    brdf_diffuse_disney,
    # Specular BRDF
    brdf_specular,
    # Combined BRDF
    compute_f0,
    PBRParamsSimple,
    evaluate_brdf,
    # Reference values
    BRDF_REFERENCE_VALUES,
    BRDF_EDGE_CASES,
)

from trinity.materials.anisotropy import (
    # WGSL source access
    get_anisotropy_wgsl,
    # Parameters
    AnisotropyParams,
    # Alpha computation
    compute_aniso_alphas,
    # Tangent rotation
    rotate_tangent,
    rotate_bitangent,
    # Anisotropic NDF
    d_ggx_anisotropic,
    # Anisotropic geometry
    g1_ggx_anisotropic,
    g_smith_ggx_anisotropic,
    # Complete BRDF
    evaluate_aniso_brdf,
    # Reference values
    ANISOTROPY_REFERENCE_VALUES,
    ANISOTROPY_EDGE_CASES,
)

from trinity.materials.variant_registry import (
    CompiledVariant,
    MaterialVariantRegistry,
    select_material_variant,
    create_quality_optimized_registry,
)

from trinity.materials.clear_coat import (
    # WGSL source access
    get_clear_coat_wgsl,
    # Parameters
    ClearCoatParams,
    # Fresnel function
    f_clear_coat,
    # NDF function
    d_clear_coat,
    # Geometry functions
    g_clear_coat_kelemen,
    g_clear_coat,
    # Evaluation functions
    evaluate_clear_coat,
    evaluate_clear_coat_with_fresnel,
    # Layer combination
    combine_clear_coat,
    combine_clear_coat_simple,
    # Convenience functions
    get_clear_coat_attenuation,
    # Reference values
    CLEAR_COAT_REFERENCE_VALUES,
    CLEAR_COAT_EDGE_CASES,
    # Constants
    CLEAR_COAT_F0,
)

from trinity.materials.sss_shader import (
    # WGSL source access
    get_sss_wgsl,
    # Data structures
    SSSProfile,
    SSSParams,
    # Predefined profiles
    SSS_PROFILE_SKIN,
    SSS_PROFILE_WAX,
    SSS_PROFILE_JADE,
    SSS_PROFILE_MILK,
    # Diffusion functions
    burley_diffusion,
    burley_diffusion_rgb,
    evaluate_diffusion_profile,
    # Kernel computation
    compute_sss_kernel,
    SSS_KERNEL_WEIGHTS,
    SSS_KERNEL_OFFSETS,
    SSS_KERNEL_SIZE,
    # Application functions
    apply_sss,
    apply_sss_with_bleeding,
    # Transmission
    evaluate_sss_transmission,
    # LUT generation
    compute_diffusion_lut_value,
    generate_diffusion_lut,
    # Profile utilities
    get_diffusion_profile_samples,
    get_sss_mask,
    # Reference values
    SSS_REFERENCE_VALUES,
    SSS_EDGE_CASES,
)

from trinity.materials.lod import (
    LODConfig,
    LODBlendInfo,
    MaterialLODSelector,
    DEFAULT_LOD_DISTANCES,
    DEFAULT_LOD_QUALITIES,
    MAX_LOD_LEVELS,
    compute_blend_factor,
    smooth_blend_factor,
    create_lod_selector_with_defaults,
    create_lod_selector_for_quality,
)

from trinity.materials.sheen import (
    # WGSL source access
    get_sheen_wgsl,
    # Parameters
    SheenParams,
    # Distribution function
    d_charlie,
    d_charlie_simple,
    # Visibility functions
    v_neubelt,
    v_ashikhmin,
    # Evaluation functions
    evaluate_sheen,
    evaluate_sheen_with_NoL,
    sheen_contribution,
    combine_brdf_with_sheen,
    # Reference values
    SHEEN_REFERENCE_VALUES,
    SHEEN_EDGE_CASES,
)

from trinity.materials.iridescence import (
    # WGSL source access
    get_iridescence_wgsl,
    # Parameter class
    IridescenceParams,
    # Presets
    IRIDESCENCE_PRESETS,
    get_preset as get_iridescence_preset,
    PRESET_SOAP_BUBBLE,
    PRESET_OIL_SLICK,
    PRESET_BEETLE,
    PRESET_PEARL,
    # Fresnel functions
    snell_cos_theta_t,
    fresnel_dielectric,
    fresnel_air_film,
    fresnel_film_substrate,
    # Phase computation
    compute_film_phase,
    # Interference
    compute_interference,
    compute_interference_color,
    # Main functions
    evaluate_iridescence,
    apply_iridescence,
    # Reference values
    IRIDESCENCE_REFERENCE_VALUES,
    IRIDESCENCE_EDGE_CASES,
    # Constants
    WAVELENGTH_R,
    WAVELENGTH_G,
    WAVELENGTH_B,
)

from trinity.materials.lighting import (
    # WGSL source access
    get_lighting_wgsl,
    # Types
    LightType,
    Light,
    LightSample,
    LightingResult,
    PBRParamsLighting,
    # Attenuation functions
    attenuation_point,
    attenuation_spot_angle,
    # Light evaluation
    evaluate_directional_light,
    evaluate_point_light,
    evaluate_spot_light,
    evaluate_light,
    # Shadow sampling
    sample_shadow,
    # Lighting accumulation
    accumulate_lighting,
    compose_final_shading,
    evaluate_all_lighting,
    # Light creation
    create_directional_light,
    create_point_light,
    create_spot_light,
    # Reference values
    LIGHTING_REFERENCE_VALUES,
    LIGHTING_EDGE_CASES,
    # Constants
    MAX_LIGHTS,
)

from trinity.materials.inheritance import (
    # Exceptions
    InheritanceError,
    NoParentSurfaceError,
    CircularInheritanceError,
    DiamondConflictError,
    # Data classes
    InheritanceInfo,
    TextureMergeResult,
    SuperCallInfo,
    # Main classes
    InheritanceResolver,
    SuperCallDetector,
    MROWalker,
    # Functions
    resolve_parent_surface,
    inline_super_call,
    merge_texture_declarations,
    generate_combined_wgsl,
    validate_inheritance,
)

from trinity.materials.pipeline_integration import (
    # Types
    ColorFormat as PipelineColorFormat,
    CullMode,
    BlendMode as PipelineBlendMode,
    PipelineConfig,
    PipelineCacheHandle,
    CachedPipeline,
    # Statistics
    ShaderCacheStats,
    LruPipelineStats,
    # Caches
    ShaderCache,
    LruPipelineTable,
    # Main interface
    PipelineIntegration,
    # Utilities
    content_hash,
    shader_hash,
)

from trinity.materials.bindless import (
    # Constants
    MAX_BINDLESS_TEXTURES,
    DEFAULT_BINDLESS_CAPACITY,
    INVALID_TEXTURE_INDEX,
    MIN_TEXTURES_FOR_BINDLESS,
    # Enums
    TextureFormat as BindlessTextureFormat,
    FilterMode as BindlessFilterMode,
    AddressMode as BindlessAddressMode,
    # Data classes
    SamplerConfig,
    TextureSlot,
    BindlessCapabilities,
    BindlessArrayStats,
    # Main class
    BindlessTextureArray,
    # Factory functions
    create_bindless_array,
    create_default_slots,
)

from trinity.materials.animation import (
    # WGSL access
    get_time_wgsl,
    TIME_WGSL,
    # Core types
    TimeUniforms,
    TimeContext,
    # Animation curves
    AnimationCurve,
    WaveType,
    EasingType,
    # Animator
    MaterialAnimator,
    AnimationState,
    AnimatedParameter,
    # Reference values
    ANIMATION_REFERENCE_VALUES,
)

__all__ = [
    # Core classes
    "Material",
    "MaterialMeta",
    "SurfaceContext",
    "SurfaceOutput",
    "MaterialCompiler",
    # Texture types
    "Texture2D",
    "TextureCube",
    "TextureDescriptor",
    "TextureBindingSet",
    "FilterMode",
    "AddressMode",
    "TextureFormat",
    "DEFAULT_TEXTURES",
    "get_default_texture",
    "is_valid_default",
    "collect_texture_descriptors",
    # Vector types
    "Vec2",
    "Vec3",
    "Vec4",
    # Decorators
    "surface",
    # Utilities
    "PythonToWGSLTranslator",
    "WGSLTranslationError",
    # Builtins registry
    "BUILTIN_REGISTRY",
    "get_builtin_wgsl",
    "get_required_builtins",
    # Noise functions
    "value_noise",
    "perlin_noise",
    "simplex_noise",
    "worley_noise",
    "fbm",
    # Color conversion
    "rgb_to_hsv",
    "hsv_to_rgb",
    "linear_to_srgb",
    "srgb_to_linear",
    "tonemap_reinhard",
    "tonemap_aces",
    # Variant system
    "VariantDomain",
    "VariantBlendMode",
    "QualityTier",
    "VariantConfig",
    "VariantCompiler",
    "generate_all_variant_combinations",
    "get_variant_for_material_system",
    # Quality tier variants (T-MAT-2.4)
    "QualityFeatures",
    "QualityShaderCode",
    "get_quality_config_for_device",
    # Blend mode variants (T-MAT-2.3)
    "BlendFactor",
    "BlendOperation",
    "ColorWriteMask",
    "BlendState",
    "BlendShaderCode",
    "get_blend_state_for_variant",
    "validate_blend_combination",
    # Include system
    "IncludeResolver",
    "IncludeError",
    "CyclicIncludeError",
    "MaxDepthError",
    "IncludeFileNotFoundError",
    "IncludeDirective",
    "DepGraph",
    "preprocess_wgsl",
    # Domain variants (T-MAT-2.2)
    "DomainCapability",
    "DomainOutputFormat",
    "DomainShaderTemplate",
    "DomainVariantGenerator",
    "DOMAIN_CAPABILITIES",
    "DOMAIN_OUTPUT_FORMATS",
    "domain_has_capability",
    "get_domain_shader_info",
    # UI Material Domain (T-MAT-5.8)
    "UIBlendMode",
    "UIMaterialConfig",
    "UIMaterialBuilder",
    "UI_DOMAIN_WGSL",
    "UI_DOMAIN_MINIMAL_WGSL",
    "UI_MATERIAL_PRESETS",
    "generate_ui_material",
    "generate_ui_material_consts",
    "get_ui_entry_point",
    "get_ui_material_preset",
    "validate_ui_material_wgsl",
    # Material dependency graph (T-MAT-2.6)
    "MaterialDepGraph",
    # Hot-reload system (T-MAT-2.7)
    "HotReloadConfig",
    "HotReloadWatcher",
    "HotReloadManager",
    "CompilationResult",
    "MaterialFileHandler",
    "WATCHDOG_AVAILABLE",
    # PBR types (T-MAT-3.1)
    "PBRInput",
    "PBRParams",
    "PBROutput",
    "get_pbr_structs_wgsl",
    "PBR_STRUCTS_WGSL",
    "PBR_PARAMS_FIELDS",
    # BRDF functions (T-MAT-3.2)
    "get_brdf_wgsl",
    "d_ggx",
    "g1_schlick_ggx",
    "g_smith_ggx",
    "g_smith_schlick",
    "f_schlick",
    "f_schlick_roughness",
    "f_schlick_scalar",
    "brdf_diffuse",
    "brdf_diffuse_disney",
    "brdf_specular",
    "compute_f0",
    "PBRParamsSimple",
    "evaluate_brdf",
    "BRDF_REFERENCE_VALUES",
    "BRDF_EDGE_CASES",
    # Anisotropic BRDF (T-MAT-4.3)
    "get_anisotropy_wgsl",
    "AnisotropyParams",
    "compute_aniso_alphas",
    "rotate_tangent",
    "rotate_bitangent",
    "d_ggx_anisotropic",
    "g1_ggx_anisotropic",
    "g_smith_ggx_anisotropic",
    "evaluate_aniso_brdf",
    "ANISOTROPY_REFERENCE_VALUES",
    "ANISOTROPY_EDGE_CASES",
    # Variant Registry (T-MAT-5.1)
    "CompiledVariant",
    "MaterialVariantRegistry",
    "select_material_variant",
    "create_quality_optimized_registry",
    # LOD System (T-MAT-5.6)
    "LODConfig",
    "LODBlendInfo",
    "MaterialLODSelector",
    "DEFAULT_LOD_DISTANCES",
    "DEFAULT_LOD_QUALITIES",
    "MAX_LOD_LEVELS",
    "compute_blend_factor",
    "smooth_blend_factor",
    "create_lod_selector_with_defaults",
    "create_lod_selector_for_quality",
    # Clear Coat BRDF (T-MAT-4.2)
    "get_clear_coat_wgsl",
    "ClearCoatParams",
    "f_clear_coat",
    "d_clear_coat",
    "g_clear_coat_kelemen",
    "g_clear_coat",
    "evaluate_clear_coat",
    "evaluate_clear_coat_with_fresnel",
    "combine_clear_coat",
    "combine_clear_coat_simple",
    "get_clear_coat_attenuation",
    "CLEAR_COAT_REFERENCE_VALUES",
    "CLEAR_COAT_EDGE_CASES",
    "CLEAR_COAT_F0",
    # Subsurface Scattering (T-MAT-4.1)
    "get_sss_wgsl",
    "SSSProfile",
    "SSSParams",
    "SSS_PROFILE_SKIN",
    "SSS_PROFILE_WAX",
    "SSS_PROFILE_JADE",
    "SSS_PROFILE_MILK",
    "burley_diffusion",
    "burley_diffusion_rgb",
    "evaluate_diffusion_profile",
    "compute_sss_kernel",
    "SSS_KERNEL_WEIGHTS",
    "SSS_KERNEL_OFFSETS",
    "SSS_KERNEL_SIZE",
    "apply_sss",
    "apply_sss_with_bleeding",
    "evaluate_sss_transmission",
    "compute_diffusion_lut_value",
    "generate_diffusion_lut",
    "get_diffusion_profile_samples",
    "get_sss_mask",
    "SSS_REFERENCE_VALUES",
    "SSS_EDGE_CASES",
    # Sheen BRDF (T-MAT-4.4)
    "get_sheen_wgsl",
    "SheenParams",
    "d_charlie",
    "d_charlie_simple",
    "v_neubelt",
    "v_ashikhmin",
    "evaluate_sheen",
    "evaluate_sheen_with_NoL",
    "sheen_contribution",
    "combine_brdf_with_sheen",
    "SHEEN_REFERENCE_VALUES",
    "SHEEN_EDGE_CASES",
    # Iridescence (T-MAT-4.6)
    "get_iridescence_wgsl",
    "IridescenceParams",
    "IRIDESCENCE_PRESETS",
    "get_iridescence_preset",
    "PRESET_SOAP_BUBBLE",
    "PRESET_OIL_SLICK",
    "PRESET_BEETLE",
    "PRESET_PEARL",
    "snell_cos_theta_t",
    "fresnel_dielectric",
    "fresnel_air_film",
    "fresnel_film_substrate",
    "compute_film_phase",
    "compute_interference",
    "compute_interference_color",
    "evaluate_iridescence",
    "apply_iridescence",
    "IRIDESCENCE_REFERENCE_VALUES",
    "IRIDESCENCE_EDGE_CASES",
    "WAVELENGTH_R",
    "WAVELENGTH_G",
    "WAVELENGTH_B",
    # Lighting functions (T-MAT-3.3)
    "get_lighting_wgsl",
    "LightType",
    "Light",
    "LightSample",
    "LightingResult",
    "PBRParamsLighting",
    "attenuation_point",
    "attenuation_spot_angle",
    "evaluate_directional_light",
    "evaluate_point_light",
    "evaluate_spot_light",
    "evaluate_light",
    "sample_shadow",
    "accumulate_lighting",
    "compose_final_shading",
    "evaluate_all_lighting",
    "create_directional_light",
    "create_point_light",
    "create_spot_light",
    "LIGHTING_REFERENCE_VALUES",
    "LIGHTING_EDGE_CASES",
    "MAX_LIGHTS",
    # Inheritance (T-MAT-5.2)
    "InheritanceError",
    "NoParentSurfaceError",
    "CircularInheritanceError",
    "DiamondConflictError",
    "InheritanceInfo",
    "TextureMergeResult",
    "SuperCallInfo",
    "InheritanceResolver",
    "SuperCallDetector",
    "MROWalker",
    "resolve_parent_surface",
    "inline_super_call",
    "merge_texture_declarations",
    "generate_combined_wgsl",
    "validate_inheritance",
    # Pipeline Integration (T-MAT-3.4)
    "PipelineColorFormat",
    "CullMode",
    "PipelineBlendMode",
    "PipelineConfig",
    "PipelineCacheHandle",
    "CachedPipeline",
    "ShaderCacheStats",
    "LruPipelineStats",
    "ShaderCache",
    "LruPipelineTable",
    "PipelineIntegration",
    "content_hash",
    "shader_hash",
    # Bindless Texture Arrays (T-MAT-5.7)
    "MAX_BINDLESS_TEXTURES",
    "DEFAULT_BINDLESS_CAPACITY",
    "INVALID_TEXTURE_INDEX",
    "MIN_TEXTURES_FOR_BINDLESS",
    "BindlessTextureFormat",
    "BindlessFilterMode",
    "BindlessAddressMode",
    "SamplerConfig",
    "TextureSlot",
    "BindlessCapabilities",
    "BindlessArrayStats",
    "BindlessTextureArray",
    "create_bindless_array",
    "create_default_slots",
    # Animation System (T-MAT-5.5)
    "get_time_wgsl",
    "TIME_WGSL",
    "TimeUniforms",
    "TimeContext",
    "AnimationCurve",
    "WaveType",
    "EasingType",
    "MaterialAnimator",
    "AnimationState",
    "AnimatedParameter",
    "ANIMATION_REFERENCE_VALUES",
]
