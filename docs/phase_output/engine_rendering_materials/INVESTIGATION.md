# Investigation: engine/rendering/materials

## Summary
The materials system is a REAL IMPLEMENTATION with comprehensive PBR, material graphs, shader compilation infrastructure, and advanced shading models. It generates actual GLSL shader code from the material graph system (not WGSL), with full support for Fresnel, normal blending, parallax mapping, noise functions, and more. The ShaderCompiler has placeholder bytecode generation but includes complete infrastructure for permutation management, PSO caching, and hot-reload.

## Files
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `__init__.py` | 277 | REAL | Full re-exports, comprehensive API |
| `material_system.py` | 904 | REAL | MaterialTemplate, MaterialInstance, MaterialFunction, MaterialLayer, MaterialSystem with dirty tracking |
| `shader_compiler.py` | 902 | PARTIAL | Full permutation/PSO infrastructure, placeholder `_compile_internal` |
| `material_graph.py` | 1281 | REAL | Node graph system generates actual GLSL code |
| `material_functions.py` | 1091 | REAL | 20+ reusable shader functions with real GLSL code |
| `pbr_model.py` | 698 | REAL | PBRParameters, PBRMaterial, PBRTextureSet with dirty flags |
| `advanced_models.py` | 636 | REAL | SSS, ClearCoat, Anisotropy, Sheen, Iridescence, Transmission |
| `constants.py` | ~100 | REAL | PBR parameter ranges, cache sizes |

## Material System Components

### Core Material System (`material_system.py`)
- `MaterialTemplate`: Base shader definition with parameter schema
- `MaterialInstance`: Override parameters for a template, weak references
- `MaterialFunction`: Reusable shader snippets with dependency tracking
- `MaterialLayer`: Composable material stacking with blend settings
- `MaterialSystem`: Central registry with hot-reload support
- `DirtyFlags`: GPU re-upload tracking (parameters, textures, shader)

### PBR Model (`pbr_model.py`)
- `PBRParameters`: Core metallic-roughness dataclass with validation
- `PBRMaterial`: Component with tracked descriptors, dirty flags, callbacks
- `PBRTextureSet`: Texture bindings with channel configuration
- `PBRWorkflow`: Metallic-roughness or specular-glossiness

### Material Graph (`material_graph.py`)
- `MaterialNode`: Abstract base with ports, code generation
- Math nodes: Add, Subtract, Multiply, Divide, Lerp, Clamp, Power, etc.
- Texture nodes: TextureSampleNode (with rgba/rgb/r/g/b/a outputs), UVNode
- Utility nodes: OneMinus, ComponentMask, AppendNode
- `OutputNode`: PBR outputs (base_color, metallic, roughness, normal, emissive, ao, opacity)
- `MaterialGraph`: DAG container with validation, cycle detection, topological sort
- `GraphCompiler`: Compiles graph to GLSL with uniform/sampler declarations

### Shader Compiler (`shader_compiler.py`)
- `ShaderSource`: Load from file or string, language auto-detection
- `ShaderStage`: vertex, fragment, compute, geometry, tessellation, mesh, raytracing
- `ShaderLanguage`: HLSL, GLSL, Metal, SPIRV, WGSL (enum only)
- `ShaderPermutation`: Feature combinations with conflict detection
- `PermutationKey`: Frozenset-based variant selection
- `CompiledShader`: Bytecode, reflection data, timing
- `PSOCache`: LRU cache with hit/miss stats
- `HotReloadWatcher`: File change polling

### Material Functions (`material_functions.py`)
- Fresnel, FresnelSchlick, FresnelSchlickRoughness
- NormalBlend (whiteout), NormalBlendRNM (reoriented normal mapping)
- ParallaxOffset, ParallaxOcclusionMapping
- TriplanarSample
- sRGB/Linear conversions, Luminance, Saturation, Contrast
- ValueNoise, Voronoi, GradientNoise
- Checkerboard, RadialGradient
- BoxMask, SphereMask
- BlendOverlay, BlendSoftLight

### Advanced Models (`advanced_models.py`)
- `SubsurfaceScattering`: Burley diffusion profiles, presets (Skin, Wax, Jade, Milk)
- `ClearCoat`: Secondary specular with IOR-based F0
- `Anisotropy`: Directional roughness with GGX parameterization
- `Sheen`: Fabric/velvet effects
- `Iridescence`: Thin film interference with thickness maps
- `Transmission`: Glass/refraction with Beer-Lambert attenuation

## Shader Generation

### Generates WGSL?
**NO** - The code generates GLSL, not WGSL. The enum includes `ShaderLanguage.WGSL` but `GraphCompiler` outputs GLSL syntax.

### Generates GLSL?
**YES** - Full GLSL code generation from:
1. Material graph nodes (each node has `generate_code` producing GLSL)
2. Material functions library (embedded GLSL strings)

### Just data structures?
**NO** - This is functional code with:
- Real GLSL shader snippets in material functions
- Node-to-code generation in GraphCompiler
- Complete material pipeline infrastructure

## Verdict
**REAL IMPLEMENTATION**

The materials system is production-quality Python code with:
- Complete PBR metallic-roughness workflow
- Node-based material authoring with real code generation
- 20+ material functions with working GLSL
- Advanced shading models (SSS, clearcoat, anisotropy, etc.)
- Shader permutation/variant management
- PSO caching infrastructure

The only placeholder is `ShaderCompiler._compile_internal()` which returns a hash instead of actual bytecode. This would need integration with glslang/dxc/naga for real compilation.

## Evidence

### GraphCompiler generates real GLSL (material_graph.py:1170-1252)
```python
def compile(self, graph: MaterialGraph) -> str:
    lines: List[str] = []
    # Generate uniform declarations
    lines.append("// Uniforms")
    for param in graph.get_parameters():
        glsl_type = type_map.get(param._data_type, "float")
        lines.append(f"uniform {glsl_type} u_{param.param_name};")
    # Generate sampler declarations
    lines.append("\n// Samplers")
    for tex in graph.get_textures():
        lines.append(f"uniform sampler2D tex_{tex.texture_name};")
    lines.append("\n// Main shader code")
    lines.append("void materialMain() {")
    # ... node code generation in topological order
```

### Material function with real GLSL (material_functions.py:110-150)
```python
def create_fresnel_function() -> MaterialFunction:
    code = """
// Fresnel effect using Schlick approximation
float Fresnel(vec3 viewDir, vec3 normal, float power) {
    float NdotV = max(dot(normal, viewDir), 0.0);
    return pow(1.0 - NdotV, power);
}
"""
```

### Node code generation example (material_graph.py:381-395)
```python
class TextureSampleNode(MaterialNode):
    def generate_code(self, input_vars, output_var):
        uv = input_vars.get("uv", "v_uv")
        lines = [
            f"vec4 {output_var}_rgba = texture(tex_{self._texture_name}, {uv});",
            f"vec3 {output_var}_rgb = {output_var}_rgba.rgb;",
            # ... channel extraction
        ]
        return "\n".join(lines)
```

### ShaderCompiler placeholder (shader_compiler.py:869-881)
```python
def _compile_internal(self, source: ShaderSource, optimize: bool) -> bytes:
    """Internal compilation implementation.
    This is a placeholder that should be overridden or extended
    with actual compiler integration (glslang, dxc, etc.).
    """
    # Placeholder: return hash of source as "bytecode"
    code = source.get_preprocessed_code()
    return hashlib.sha256(code.encode()).digest()
```
