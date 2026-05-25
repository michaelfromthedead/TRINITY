# SUMMARY: engine/rendering/materials

## Metrics Table

| Metric | Value |
|--------|-------|
| **Total Files** | 8 |
| **Total Lines** | 5,976 |
| **Implementation Status** | 95% REAL |
| **Placeholder Code** | 1 function (`_compile_internal`) |
| **Public API Surface** | 50+ exports |
| **Material Functions** | 20+ |
| **Node Types** | 25+ |
| **Shading Models** | 6 advanced |

### File Breakdown

| File | Lines | Bytes | Status |
|------|-------|-------|--------|
| `material_graph.py` | 1,280 | 36,913 | REAL |
| `material_functions.py` | 1,090 | 32,334 | REAL |
| `material_system.py` | 903 | 27,900 | REAL |
| `shader_compiler.py` | 901 | 27,241 | PARTIAL |
| `pbr_model.py` | 697 | 21,273 | REAL |
| `advanced_models.py` | 635 | 21,014 | REAL |
| `__init__.py` | 276 | 6,486 | REAL |
| `constants.py` | 194 | 6,423 | REAL |

## Algorithm Inventory

| Algorithm | File | Status | Description |
|-----------|------|--------|-------------|
| Graph Topological Sort | `material_graph.py` | REAL | Order nodes for code generation |
| Cycle Detection | `material_graph.py` | REAL | Validate DAG structure |
| Type Promotion | `material_graph.py` | REAL | Float-to-vector coercion |
| GLSL Code Generation | `material_graph.py` | REAL | Node-to-shader compilation |
| Fresnel (Schlick) | `material_functions.py` | REAL | View-dependent reflectance |
| Normal Blending (Whiteout) | `material_functions.py` | REAL | Detail normal compositing |
| Normal Blending (RNM) | `material_functions.py` | REAL | Reoriented normal mapping |
| Parallax Offset | `material_functions.py` | REAL | Simple height mapping |
| Parallax Occlusion Mapping | `material_functions.py` | REAL | Raymarched height mapping |
| Triplanar Sampling | `material_functions.py` | REAL | World-space texture projection |
| sRGB Conversion | `material_functions.py` | REAL | Linear-sRGB color space |
| Value Noise | `material_functions.py` | REAL | Procedural noise generation |
| Voronoi | `material_functions.py` | REAL | Cellular noise patterns |
| Gradient Noise | `material_functions.py` | REAL | Perlin-style noise |
| Burley Diffusion | `advanced_models.py` | REAL | SSS profile sampling |
| Thin Film Interference | `advanced_models.py` | REAL | Iridescence calculation |
| Beer-Lambert Attenuation | `advanced_models.py` | REAL | Transmission absorption |
| GGX Anisotropy | `advanced_models.py` | REAL | Directional roughness |
| Permutation Hashing | `shader_compiler.py` | REAL | Variant key generation |
| LRU PSO Cache | `shader_compiler.py` | REAL | Pipeline state caching |
| Bytecode Compilation | `shader_compiler.py` | PLACEHOLDER | Returns hash, not bytecode |
| Hot-Reload Polling | `shader_compiler.py` | REAL | File change detection |

## Evidence Snippets

### 1. GLSL Code Generation (material_graph.py:1170-1200)
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
```

### 2. Fresnel Function with Real GLSL (material_functions.py:110-130)
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

### 3. Texture Sample Node Code Generation (material_graph.py:381-395)
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

### 4. Burley SSS Profile (advanced_models.py:67-94)
```python
def get_diffusion_profile(self, num_samples: int = 16) -> List[float]:
    d = self.scatter_radius
    samples = []
    for i in range(num_samples):
        r = (i + 0.5) * (d * 3.0) / num_samples
        # Burley normalized diffusion
        weight = (
            math.exp(-r / d) / (2.0 * math.pi * d * d)
            + math.exp(-r / (3.0 * d)) / (6.0 * math.pi * d * d)
        )
        samples.append(weight)
```

### 5. Placeholder Compilation (shader_compiler.py:869-881)
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

### 6. Material Dirty Flags (material_system.py)
```python
class DirtyFlags(enum.Flag):
    NONE = 0
    PARAMETERS = enum.auto()
    TEXTURES = enum.auto()
    SHADER = enum.auto()
    ALL = PARAMETERS | TEXTURES | SHADER
```

## Dependency Graph

```
engine.core.math.vec (Vec2, Vec3, Vec4)
         |
         v
+--------+--------+
|                 |
v                 v
material_system   pbr_model
|                 |
+--------+--------+
         |
         v
   material_graph
         |
         v
 material_functions
         |
         v
  advanced_models
         |
         v
  shader_compiler
```
