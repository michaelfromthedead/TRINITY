from .ast_nodes import (
    Axis, BendNode, BoxNode, BoxFrameNode, CameraNode, CapsuleNode, CellIdNode,
    CombineNode, CompensationNode, ConeNode, CylinderNode, DomainOpNode,
    EllipsoidNode, ExprNode, FloatNode, FullSceneNode, IntersectionNode,
    KifsNode, Kind, LightNode, LightType, MaterialNode, MirrorNode,
    OctahedronNode, PlaneNode, PositionNode, PyramidNode, RenderSettingsNode,
    RepeatNode, RoundedBoxNode, SceneGraph, SdfPrimitiveNode, SphereNode,
    StretchNode, SubtractionNode, TorusNode, TwistNode, UnionNode, Vec3Node,
    SDF_PRIMITIVE_TYPE_MAP, DOMAIN_OP_TYPE_MAP,
)
from .ast_builder import (
    AstBuilder, walk_composition, build_from_composition,
)
from .wgsl_codegen import (
    WgslCodeGen, generate_wgsl, generate_wgsl_from_scene,
    GENERATED_HEADER,
)
from .sdf_codegen import (
    WGSLCodegen,
    TransformContext,
    generate_primitive_wgsl,
    generate_domain_op_wgsl,
    generate_scene_sdf,
    get_all_primitive_wgsl,
    get_all_domain_op_wgsl,
    PRIMITIVE_WGSL_FUNCTIONS,
    DOMAIN_OP_WGSL_FUNCTIONS,
)
from .material_codegen import (
    MaterialCodegen, generate_material_wgsl, MATERIAL_STRUCT,
)
from .sdf_optimizer import (
    OptimizationPass,
    ConstantFoldingPass,
    DeadCodeEliminationPass,
    CommonSubexpressionEliminationPass,
    DomainRepetitionFlatteningPass,
    MaterialMergingPass,
    SDFOptimizer,
    DEFAULT_PASSES,
    FAST_PASSES,
    AGGRESSIVE_PASSES,
    optimize_ast,
    fold_constants,
    eliminate_dead_code,
    eliminate_common_subexpressions,
    flatten_repeats,
    merge_materials,
    ast_hash,
    ast_equal,
)
from .scene_codegen import (
    SceneCodegen, generate_scene_wgsl, generate_compute_shader,
)

# T-DEMO-2.1 / T-DEMO-2.2: Trinity-enhanced SDF AST with Mirror/Tracker patterns
from .sdf_ast import (
    # Base classes
    SDFNode as TrinitySDFNode,
    SDFNodeMeta,
    # Vec3 helper
    Vec3 as TrinityVec3,
    Axis as TrinityAxis,
    # Primitive nodes
    PrimitiveNode as TrinityPrimitiveNode,
    SphereNode as TrinitySphereNode,
    BoxNode as TrinityBoxNode,
    TorusNode as TrinityTorusNode,
    CylinderNode as TrinityCylinderNode,
    ConeNode as TrinityConeNode,
    PlaneNode as TrinityPlaneNode,
    CapsuleNode as TrinityCapsuleNode,
    EllipsoidNode as TrinityEllipsoidNode,
    BoxFrameNode as TrinityBoxFrameNode,
    RoundedBoxNode as TrinityRoundedBoxNode,
    OctahedronNode as TrinityOctahedronNode,
    PyramidNode as TrinityPyramidNode,
    # Combinator nodes
    CombinatorNode as TrinityCombinatorNode,
    UnionNode as TrinityUnionNode,
    IntersectionNode as TrinityIntersectionNode,
    SubtractionNode as TrinitySubtractionNode,
    SmoothUnionNode as TrinitySmoothUnionNode,
    SmoothIntersectionNode as TrinitySmoothIntersectionNode,
    SmoothSubtractionNode as TrinitySmoothSubtractionNode,
    DisplacedNode as TrinityDisplacedNode,
    # Domain operation nodes
    DomainOpNode as TrinityDomainOpNode,
    RepeatNode as TrinityRepeatNode,
    MirrorNode as TrinityMirrorNode,
    KIFSNode as TrinityKIFSNode,
    TwistNode as TrinityTwistNode,
    BendNode as TrinityBendNode,
    StretchNode as TrinityStretchNode,
    # Scene nodes
    MaterialNode as TrinityMaterialNode,
    SceneNode as TrinitySceneNode,
    CameraNode as TrinityCameraNode,
    LightNode as TrinityLightNode,
    RenderSettingsNode as TrinityRenderSettingsNode,
    # Trinity pattern helpers
    Mirror,
    Tracker,
    # AST builder
    build_ast as trinity_build_ast,
)

# T-DEMO-2.13: Cached Compilation with Tracker Dirty Invalidation
from .sdf_cache import (
    # Cache classes
    WGSLCache,
    CachedSDFCompiler,
    CacheStats,
    CacheEntry,
    OptimizationLevel,
    # Hash functions
    sdf_node_hash,
    # Validity checking
    is_cache_valid,
    # Factory functions
    create_cached_compiler,
)

# T-DEMO-2.14: SDF Error Reporting
from .sdf_errors import (
    # Exception hierarchy
    SDFError,
    SDFValidationError,
    SDFCompilationError,
    SDFTypeError,
    SDFRecursionError,
    SDFImpossibleSDFError,
    # Validation
    SDFValidator,
    ValidationIssue,
    Severity,
    # Convenience functions
    validate_scene,
    validate_scene_strict,
    is_scene_valid,
    get_validation_report,
    # Compiler integration
    ValidatingCompiler,
)

# T-DEMO-3.1: Camera Ray Generation (Pinhole Model)
from .ray_generation import (
    Ray,
    RayGenerator,
    Vec3 as RayVec3,
    CameraParams,
    generate_ray_wgsl,
    generate_ray_wgsl_inline,
    validate_camera,
)

# T-DEMO-3.2, T-DEMO-3.3, T-DEMO-3.4: Ray Marching with Sphere Tracing
from .ray_march import (
    # T-DEMO-3.2: Ray Marching Loop (Sphere Tracing)
    HitResult,
    MarchResultType,
    SphereTracer,
    march_ray,
    generate_ray_march_struct_wgsl,
    # T-DEMO-3.3: Perceptual Termination Criterion
    epsilon_at_distance,
    PerceptualEpsilonConfig,
    RayMarchConfig,
    RayMarcher,
    RayMarchResult,
    # T-DEMO-3.4: Normal Estimation
    estimate_normal,
    NormalEstimationConfig,
    NormalEstimator,
    # WGSL Generation
    generate_epsilon_wgsl,
    generate_normal_estimation_wgsl,
    generate_ray_march_wgsl,
    # Reference SDF primitives
    sdf_sphere,
    sdf_box,
    sdf_plane,
    sdf_torus,
    sdf_cylinder,
)

# T-DEMO-3.5: SDF Ambient Occlusion (Quilez's Method)
from .sdf_ao import (
    # Configuration
    AOConfig,
    # Core functions
    calculate_ao,
    calculate_ao_multi_direction,
    # WGSL generation
    AO_WGSL_FUNCTION,
    generate_ao_wgsl,
    generate_ao_wgsl_inline,
    # Helpers
    make_scene_ao_evaluator,
)

# T-DEMO-3.6: Soft SDF Shadows (Quilez's Penumbra)
from .sdf_shadows import (
    # Configuration
    ShadowConfig,
    # Core functions
    calculate_soft_shadow,
    calculate_soft_shadow_improved,
    calculate_hard_shadow,
    calculate_shadow_from_light,
    # WGSL generation
    SHADOW_WGSL_FUNCTION,
    generate_shadow_wgsl,
    generate_shadow_wgsl_improved,
    generate_shadow_wgsl_inline,
    # Helpers
    make_scene_shadow_evaluator,
    make_light_shadow_evaluator,
)

# T-DEMO-3.7 / T-DEMO-3.8: Diffuse and Specular Lighting
from .sdf_lighting import (
    # T-DEMO-3.7: Diffuse Lighting
    calculate_diffuse,
    calculate_diffuse_directional,
    calculate_all_diffuse,
    # T-DEMO-3.8: Specular Lighting (Blinn-Phong)
    calculate_specular_blinn_phong,
    calculate_specular_blinn_phong_directional,
    roughness_to_shininess,
    # T-DEMO-3.8: Specular Lighting (GGX/Cook-Torrance)
    calculate_specular_ggx,
    calculate_specular_ggx_directional,
    fresnel_schlick,
    distribution_ggx,
    geometry_schlick_ggx,
    geometry_smith,
    # Combined lighting
    calculate_lighting,
    # Data structures
    LightParams,
    MaterialParams,
    # Shadow helpers
    calculate_soft_shadow as lighting_soft_shadow,
    calculate_hard_shadow as lighting_hard_shadow,
    calculate_attenuation_inverse_square,
    calculate_attenuation_linear,
    # WGSL code generation
    LightingCodegen,
    WGSL_SOFT_SHADOW,
    WGSL_ATTENUATION,
    WGSL_DIFFUSE,
    WGSL_SPECULAR_BLINN_PHONG,
    WGSL_SPECULAR_GGX,
)

# T-DEMO-3.13: Temporal Anti-Aliasing via Sub-Pixel Jitter
from .temporal_aa import (
    # Halton sequence
    halton_sequence,
    halton_2d,
    # Jitter
    JitterPattern,
    JitterSequence,
    get_jitter,
    # Texture (for CPU-side accumulation)
    Texture as TAATexture,
    # Accumulator
    AccumulatorConfig,
    TemporalAccumulator,
    # WGSL generation
    generate_jitter_wgsl,
    generate_accumulation_wgsl,
    generate_taa_pipeline_wgsl,
)

# T-DEMO-3.1: Ray Generation
from .ray_generation import (
    Ray,
    RayGenerator,
    Vec3 as RayVec3,
    CameraParams,
    generate_ray_wgsl,
    generate_ray_wgsl_inline,
    validate_camera,
)

# T-DEMO-3.9: Full-Screen Compute Shader Dispatch
from .compute_dispatch import (
    ComputeDispatch,
    BindGroupConfig,
    OutputFormat,
    generate_entry_point_template,
    calculate_dispatch_dimensions,
)

# T-DEMO-3.10: Sky Color Functions
from .sky import (
    SkyMode,
    Vec3 as SkyVec3,
    SkyConfig,
    SkySettingsNode,
    sky_solid,
    sky_gradient,
    sky_gradient_triple,
    sky_procedural,
    generate_sky_wgsl,
    create_sunset_sky,
    create_daytime_sky,
    create_night_sky,
)

# T-DEMO-3.11: Tone Mapping for Output
from .tone_mapping import (
    # Operators enum
    ToneMappingOperator,
    # Tone mapping functions
    reinhard,
    reinhard_extended,
    aces_filmic,
    uncharted2,
    linear_clamp,
    # Gamma correction
    gamma_correct,
    linear_to_srgb,
    srgb_to_linear,
    # ToneMapper class
    ToneMapper,
    # WGSL generation
    generate_tone_mapping_wgsl,
    # Validation
    validate_color_range,
    is_valid_hdr_color,
)

# T-DEMO-4.9: Planet SDF (Spherical Terrain)
from .planet_sdf import (
    PlanetSDF,
    PlanetConfig,
    CraterConfig,
    SphericalCoord,
)

# T-DEMO-3.12: Depth of Field (Lens Jitter)
from .depth_of_field import (
    # Parameters
    DOFParams,
    # Circle of confusion
    calculate_coc,
    calculate_coc_normalized,
    is_in_focus,
    # Lens sampling
    sample_disk_uniform,
    sample_disk_stratified,
    sample_hexagon,
    # DOF Generator
    DOFGenerator,
    # Accumulation
    AccumulationBuffer,
    # WGSL generation
    generate_dof_wgsl,
    # Validation
    validate_dof_params,
)

# T-DEMO-4.10, T-DEMO-4.11: Fractal SDFs (Mandelbulb and KIFS)
from .fractal_sdf import (
    # T-DEMO-4.10: Mandelbulb SDF
    MandelbulbSDF,
    MandelbulbConfig,
    mandelbulb_distance,
    mandelbulb_distance_estimator,
    cartesian_to_spherical,
    spherical_to_cartesian,
    spherical_power,
    generate_mandelbulb_wgsl,
    MANDELBULB_WGSL_FUNCTION,
    # T-DEMO-4.11: KIFS SDF
    KIFSSDF,
    KIFSConfig,
    KIFSFoldType,
    kifs_distance,
    kifs_fold_abs,
    kifs_fold_menger,
    kifs_fold_sierpinski,
    kifs_fold_box,
    kifs_fold_sphere,
    kifs_iteration,
    generate_kifs_wgsl,
    KIFS_WGSL_FUNCTION,
    KIFS_FOLD_ABS_WGSL,
    KIFS_FOLD_SIERPINSKI_WGSL,
    KIFS_FOLD_BOX_WGSL,
    KIFS_FOLD_SPHERE_WGSL,
    # Utilities
    FractalSDFNode,
)

# T-DEMO-4.12, T-DEMO-4.13: Surface Detail (Bump Mapping and Curvature Detection)
from .surface_detail import (
    # T-DEMO-4.12: Bump Mapping
    BumpMapConfig,
    BumpMapper,
    compute_bump_normal,
    compute_noise_gradient_3d,
    # T-DEMO-4.13: Curvature Detection
    CurvatureConfig,
    CurvatureDetector,
    CurvatureResult,
    CurvatureType,
    compute_laplacian,
    detect_edges,
    detect_ridges,
    # WGSL Generation
    generate_bump_mapping_wgsl,
    generate_curvature_wgsl,
    # Noise functions
    fbm_3d,
    value_noise_3d,
    perlin_noise_3d,
)

# T-DEMO-4.14, T-DEMO-4.15, T-DEMO-4.16: Procedural Palettes and Color LUT
from .procedural_palette import (
    # T-DEMO-4.14: Height-Based Terrain Color Palettes
    TerrainPalette,
    TerrainZone,
    DEFAULT_TERRAIN_ZONES,
    # T-DEMO-4.15: Procedural Palette Patterns
    ProceduralPattern,
    PatternType,
    # T-DEMO-4.16: 256-Entry Palette LUT
    PaletteLUT,
    MaterialPaletteMap,
    # Helpers
    Vec3 as PaletteVec3,
    # WGSL generation
    generate_palette_wgsl,
    FBM_WGSL,
)

# T-DEMO-4.1, T-DEMO-4.2: Terrain SDF Functions (Heightmap and Ridged)
from .terrain_sdf import (
    # Core classes
    TerrainSDF,
    HeightmapTerrainSDF,
    RidgedTerrainSDF,
    # Configuration
    HeightmapConfig,
    RidgedConfig,
    # Helpers
    Vec3 as TerrainVec3,
    TerrainMirror,
    TerrainTracker,
    # Factory functions
    create_heightmap_terrain,
    create_ridged_terrain,
    # WGSL generation
    generate_heightmap_terrain_wgsl,
    generate_ridged_terrain_wgsl,
    generate_all_terrain_wgsl,
    # Noise functions
    fbm_2d,
    ridged_fbm_2d,
)

# T-DEMO-4.7, T-DEMO-4.8: Architecture SDF (Building and City Block)
from .architecture_sdf import (
    # Core classes
    BuildingSDF,
    CityBlockSDF,
    # Roof styles
    RoofStyle,
    # Hash functions for procedural variation
    cell_hash,
    hash_to_float,
)

# T-DEMO-4.3, T-DEMO-4.4: Advanced Terrain SDF (Domain-Warped and Cave Terrain)
from .terrain_advanced import (
    # T-DEMO-4.3: Domain-Warped Terrain
    DomainWarpedTerrainSDF,
    DomainWarpConfig,
    WarpPass,
    # T-DEMO-4.4: 3D Cave Terrain
    CaveTerrainSDF,
    CaveConfig,
    # Shared
    NoiseType,
    TerrainConfig,
    # Factory functions
    create_domain_warped_terrain,
    create_cave_terrain,
)

# T-DEMO-4.5, T-DEMO-4.6: Vegetation SDF (Tree and Forest)
from .vegetation_sdf import (
    # Enums
    TrunkType,
    CanopyType,
    # T-DEMO-4.5: Tree SDF Configuration
    TreeConfig,
    BranchConfig,
    TreeSDF,
    # T-DEMO-4.6: Forest SDF Configuration
    ForestConfig,
    TreeVariation,
    ForestSDF,
    # Hash functions for procedural variation
    cell_hash as vegetation_cell_hash,
    cell_hash_float,
    hash_to_float as vegetation_hash_to_float,
    # WGSL generation
    generate_tree_wgsl,
    generate_forest_wgsl,
    # Python SDF functions (for testing/CPU evaluation)
    sdf_tree,
    sdf_forest,
)

# T-DEMO-8.4: Importance-Driven SDF Evaluation (Adaptive Ray Marching)
from .adaptive_march import (
    # Complexity analysis
    ComplexityLevel,
    ComplexityEstimate,
    GradientAnalyzer,
    # Adaptive marching
    AdaptiveMarchConfig,
    AdaptiveMarchResult,
    AdaptiveMarcher,
    # Step scaling
    StepScaler,
    GradientBasedScaler,
    DistanceBasedScaler,
    CombinedScaler,
    # Complexity map
    ComplexityMap,
    ComplexityMapConfig,
    ComplexityMapGenerator,
    # WGSL generation
    generate_gradient_magnitude_wgsl,
    generate_adaptive_march_wgsl,
    generate_complexity_map_wgsl,
    generate_step_scaler_wgsl,
    # Convenience functions
    estimate_complexity,
    compute_gradient_magnitude,
    adaptive_march_ray,
    create_adaptive_marcher,
)

# T-DEMO-8.5: TAA for Ray Marching (Reprojection)
from .taa_reprojection import (
    # Configuration
    ReprojectionConfig,
    DisocclusionMode,
    ClampingMode,
    # Buffers
    HitPositionBuffer,
    ColorBuffer as TAAColorBuffer,
    # Color space conversion
    rgb_to_ycocg,
    ycocg_to_rgb,
    # Reprojection functions
    project_to_screen,
    calculate_reprojected_uv,
    detect_disocclusion_depth,
    detect_disocclusion_position,
    # Clamping functions
    compute_neighborhood_bounds_rgb,
    compute_neighborhood_bounds_ycocg,
    compute_neighborhood_variance,
    clamp_color_rgb,
    clamp_color_ycocg,
    clamp_color_variance,
    # Main class
    TAAReprojection,
    # WGSL generation
    generate_reprojection_wgsl,
    generate_ycocg_wgsl,
    generate_neighborhood_clamping_wgsl,
    generate_disocclusion_wgsl,
    generate_taa_reprojection_wgsl,
)

# T-DEMO-8.1: Analytic Gradient Propagation for SDF Primitives
from .analytic_gradients import (
    # Result types
    GradientResult,
    CombinatorGradientResult,
    # Primitive gradient functions (12 total)
    gradient_sphere,
    gradient_box,
    gradient_torus,
    gradient_cylinder,
    gradient_cone,
    gradient_plane,
    gradient_capsule,
    gradient_ellipsoid,
    gradient_box_frame,
    gradient_rounded_box,
    gradient_octahedron,
    gradient_pyramid,
    # Combinator gradient functions
    gradient_union,
    gradient_intersection,
    gradient_subtraction,
    gradient_smooth_union,
    gradient_smooth_intersection,
    gradient_smooth_subtraction,
    # Validation utilities
    validate_gradient,
    central_difference_gradient,
    gradient_vs_central_diff,
    normalize_gradient,
)
