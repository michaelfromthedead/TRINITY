from trinity.materials.dsl import (
    Material,
    MaterialMeta,
    PythonToWGSLTranslator,
    SurfaceContext,
    SurfaceOutput,
    Vec2,
    Vec3,
    Vec4,
    WGSLTranslationError,
    surface,
)
from trinity.materials.compiler import MaterialCompiler
from trinity.materials.textures import (
    AddressMode,
    DEFAULT_TEXTURES,
    DefaultTextureSpec,
    FilterMode,
    Texture2D,
    TextureBindingSet,
    TextureCube,
    TextureDescriptor,
    TextureFormat,
)
from trinity.materials.builtins import (
    BUILTIN_REGISTRY,
    get_builtin_wgsl,
    get_required_builtins,
)
from trinity.materials.pipeline_integration import (
    PipelineConfig,
    PipelineIntegration,
    ShaderCache,
    LruPipelineTable,
    content_hash,
    shader_hash,
)
from trinity.materials.brdf import (
    get_brdf_wgsl,
    d_ggx,
    g_smith_ggx,
    f_schlick,
    brdf_specular,
    evaluate_brdf,
    BRDF_REFERENCE_VALUES,
)

__all__ = [
    # DSL core
    "Material",
    "MaterialMeta",
    "MaterialCompiler",
    "PythonToWGSLTranslator",
    "SurfaceContext",
    "SurfaceOutput",
    "WGSLTranslationError",
    "surface",
    # Vector types
    "Vec2",
    "Vec3",
    "Vec4",
    # Texture types
    "AddressMode",
    "DEFAULT_TEXTURES",
    "DefaultTextureSpec",
    "FilterMode",
    "Texture2D",
    "TextureBindingSet",
    "TextureCube",
    "TextureDescriptor",
    "TextureFormat",
    # Builtins
    "BUILTIN_REGISTRY",
    "get_builtin_wgsl",
    "get_required_builtins",
    # Pipeline integration
    "PipelineConfig",
    "PipelineIntegration",
    "ShaderCache",
    "LruPipelineTable",
    "content_hash",
    "shader_hash",
    # BRDF functions
    "get_brdf_wgsl",
    "d_ggx",
    "g_smith_ggx",
    "f_schlick",
    "brdf_specular",
    "evaluate_brdf",
    "BRDF_REFERENCE_VALUES",
]
