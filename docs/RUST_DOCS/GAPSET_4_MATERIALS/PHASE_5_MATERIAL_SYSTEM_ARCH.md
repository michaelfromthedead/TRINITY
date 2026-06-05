# Phase 5: Material System — Architecture

## Status: PARTIAL

The Python-side material system is rich and functional (material_system.py, shader_compiler.py, material_graph.py). The Rust side has the bindless material table. WGSL domain implementations (decal, volume, UI) and material animation/LOD are absent.

## Current Architecture

### Python Material System (`engine/rendering/materials/material_system.py`)

```
MaterialSystem
├── MaterialDomain (enum): SURFACE | DEFERRED_DECAL | VOLUME | POST_PROCESS | UI
├── BlendMode (enum): OPAQUE | MASKED | TRANSLUCENT | ADDITIVE | MODULATE
├── ShadingModel (enum): UNLIT | DEFAULT_LIT | SUBSURFACE | CLEAR_COAT | CLOTH | HAIR | EYE | FOLIAGE
├── ParameterType (enum): FLOAT | VEC2 | VEC3 | VEC4 | INT | BOOL | TEXTURE_2D | TEXTURE_CUBE | SAMPLER
├── MaterialParameter { name, type, default, range, description }
├── MaterialTemplate
│   ├── name, domain, blend_mode, shading_model
│   ├── parameters: Dict[str, MaterialParameter]
│   ├── textures: Dict[str, TextureBinding]
│   ├── functions: List[MaterialFunction]
│   └── compile(source) → CompiledShader
├── MaterialInstance
│   ├── template: MaterialTemplate
│   ├── parameter_overrides: Dict[str, Any]
│   ├── texture_bindings: Dict[str, TextureBinding]
│   └── get_parameters() → merged dict
├── MaterialFunction { name, category, body, parameters, return_type }
├── MaterialLayer { name, base, blend, opacity }
├── MaterialSystem (singleton)
│   ├── templates: Dict[str, MaterialTemplate]
│   ├── instances: Dict[str, MaterialInstance]
│   ├── register_template(template)
│   ├── create_instance(template_name, overrides)
│   └── get_instance(name) → MaterialInstance
└── DirtyFlags { parameters, textures, shader }
```

### Python Shader Compiler (`engine/rendering/materials/shader_compiler.py`)

```
ShaderStage (enum): VERTEX | FRAGMENT | COMPUTE | GEOMETRY | ...
ShaderLanguage (enum): HLSL | GLSL | METAL | SPIRV | WGSL
ShaderSource { path, code, language, stage, entry_point, includes, defines }
  ├── from_file(path, stage, language) → ShaderSource
  └── from_string(code, stage, language) → ShaderSource
ShaderDefine { name, value? }
PermutationKey { defines: FrozenSet }
ShaderPermutation { defines, sources: Dict[stage, ShaderSource] }
  └── compile_variant(device) → CompiledShader
CompiledShader { vertex, fragment, hash, defines }
PSODescriptor { vertex_layout, blend_state, depth_stencil, rasterizer }
PSOCache { max_size, cache: OrderedDict }
  ├── get(key) → Option<CachedPSO>
  └── insert(key, pso)
HotReloadWatcher { poll_interval, watched_files }
```

### Python Material Graph (`engine/rendering/materials/material_graph.py`)

```
DataType (enum): FLOAT | VEC2 | VEC3 | VEC4 | INT | BOOL | TEXTURE2D | TEXTURECUBE | SAMPLER
NodePort { name, data_type, is_output, default_value, hidden }
NodeConnection { source_node, source_port, target_node, target_port }
MaterialNode (ABC) { id, inputs/outputs: List[NodePort], position }
  ├── ConstantNode    ── float/vec3/vec4 constant output
  ├── ParameterNode   ── Material parameter input
  ├── TextureSampleNode ── Texture sampling
  ├── UVNode          ── UV coordinate output
  ├── MathNode variants: Add, Subtract, Multiply, Divide, Lerp, Clamp, Dot, Normalize, Power, Sqrt, Abs, Frac, Floor, Ceil, Min, Max, Sin, Cos
  ├── OneMinus        ── 1 - input
  ├── ComponentMask   ── Channel swizzle
  ├── AppendNode      ── Vector construction
  └── OutputNode      ── Material output
MaterialGraph { nodes, connections }
  ├── add_node(node)
  ├── connect(from_node, from_port, to_node, to_port)
  └── validate() → [errors]
GraphCompiler
  └── compile(graph) → shader source code
```

### Python PBR Model (`engine/rendering/materials/pbr_model.py`)

```
PBRParameters { base_color, metallic, roughness, normal_scale, ao, emissive }
  └── validate() → (bool, [errors])
PBRMaterial (Component)
  ├── parameters: PBRParameters
  ├── textures: PBRTextureSet
  └── dirty_flags: DirtyFlags
PBRTextureSet { albedo, normal, metallic_roughness, emissive, occlusion }
PBRWorkflow: METALLIC_ROUGHNESS | SPECULAR_GLOSSINESS
```

### Python Material Functions (`engine/rendering/materials/material_functions.py`)

```
MaterialFunctionLibrary (singleton)
  ├── create_fresnel_function()     → Schlick Fresnel
  ├── create_normal_blend_function()  → Normal map blending
  ├── create_parallax_function()    → Parallax mapping
  ├── create_triplanar_function()   → Triplanar mapping
  ├── create_detail_normal_function() → Detail normal blending
  ├── create_height_blend_function()  → Height-based blending
  ├── create_srgb_to_linear_function()
  ├── create_linear_to_srgb_function()
  ├── create_luminance_function()
  ├── create_saturation_function()
  ├── create_contrast_function()
  ├── create_noise_function()
  ├── create_voronoi_function()
  ├── create_gradient_noise_function()
```

### Rust Bindless Material Table (`gpu_driven/material_table.rs`)

```
MaterialTableEntry (80 bytes, repr(C))
  ├── base_color: [f32; 4]         # offset 0
  ├── emissive: [f32; 4]           # offset 16
  ├── metallic/roughness/occlusion/normal_scale: f32  # offset 32-44
  ├── texture_ids: [u32; 4]        # offset 48-60
  ├── flags: u32                   # offset 64
  └── alpha_cutoff: f32            # offset 68

MaterialTable
  ├── entries: Vec<MaterialTableEntry>
  ├── add(entry) → u32 (index)
  ├── update(index, entry)
  ├── remove(index)
  ├── stage(registry) → SubmitResult   # Upload dirty range to GPU
  └── as_bytes() → &[u8]
```

### Rust Texture Table (`gpu_driven/texture_table.rs`)

```
TextureTable
  ├── add(desc: TextureTableEntry) → u32 (index)
  ├── update(index, desc)
  ├── remove(index) → TextureRemoveResult
  └── stage(registry) → submit GPU write
```

## Missing for Full Material System

1. **Quality-driven variant compilation** — Each material compiled 3× (low/medium/high); runtime variant selection
2. **Material inheritance (WGSL)** — `super()` calls in AST produce parent WGSL inlining; MRO resolution
3. **Decal domain WGSL** — Screen-space UV, G-buffer modification, no lighting
4. **Volume domain WGSL** — Ray-volume AABB intersection, single-scattering integral
5. **Material animation WGSL** — `time()` uniform, DSL sin(time()) modulation
6. **Material LOD** — Distance thresholds, LOD cross-fade, per-level materials
7. **Bindless texture arrays** — `binding_array<texture_2d<f32>>` in WGSL, index array
8. **UI domain WGSL** — Unlit vertex-color, fullscreen UV, no PBR

## Cross-References

- `engine/rendering/materials/material_system.py` — Core material system (27KB)
- `engine/rendering/materials/shader_compiler.py` — Shader compilation (27KB)
- `engine/rendering/materials/material_graph.py` — Node graph (37KB)
- `engine/rendering/materials/pbr_model.py` — PBR model (21KB)
- `engine/rendering/materials/material_functions.py` — Function library (32KB)
- `engine/rendering/materials/constants.py` — Constants (6KB)
- `engine/tooling/material_editor/` — Full material editor (244KB total)
- `crates/renderer-backend/src/gpu_driven/material_table.rs` — Bindless MaterialTable (45KB)
- `crates/renderer-backend/src/gpu_driven/texture_table.rs` — TextureTable (9KB)
