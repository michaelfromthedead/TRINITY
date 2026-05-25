# Phase 3: PBR Core — Architecture

## Status: MOSTLY REAL

This is the most complete phase. The WGSL PBR shaders are fully functional with Cook-Torrance BRDF, full light loop, and CSM shadows. The pipeline infrastructure (ShaderCache, PipelineTable) exists. Missing: PBR pipeline integration in the renderer (currently renders a triangle).

## Current Architecture

### PBR Fragment Shader (`shaders/pbr.frag.wgsl`)

```
pbr.frag.wgsl (377 lines)
├── MaterialTableEntry struct        # 80 bytes, bindless
│   ├── base_color: vec4<f32>
│   ├── emissive: vec4<f32>
│   ├── metallic/roughness/occlusion/normal_scale: f32
│   ├── texture_id[4]: u32
│   ├── flags: u32, alpha_cutoff: f32
├── Light types
│   ├── DirectionalLight { direction, color, intensity }
│   ├── PointLight { position, radius, color, intensity }
│   └── SpotLight { position, radius, direction, cos angles, color, intensity }
├── LightCounts { num_directional, num_point, num_spot }
├── CameraUniforms                  # Group 0
│   ├── view/projection/view_projection: mat4x4
│   └── camera_position: vec3, ambient_intensity: f32
├── Bind groups
│   ├── @group(0) camera uniforms
│   ├── @group(1) material_table (storage)
│   ├── @group(2) light buffers (uniform + storage)
│   └── @group(3) shadow maps (depth_2d_array + comparison sampler)
├── BRDF functions
│   ├── distribution_ggx(n, h, roughness) → f32              # Trowbridge-Reitz GGX NDF
│   ├── geometry_schlick_ggx(ndotv, roughness) → f32         # Smith-GGX (single)
│   ├── geometry_smith(n, v, l, roughness) → f32             # Smith-GGX (combined)
│   ├── fresnel_schlick(cos_theta, f0) → vec3<f32>           # Schlick Fresnel
│   └── eval_brdf(n, v, l, h, albedo, f0, roughness, radiance) → vec3<f32>  # Cook-Torrance
├── Shadow functions
│   ├── select_cascade(view_depth) → u32
│   └── shadow_factor(world_pos, normal, light_dir) → f32    # PCF with configurable kernel
├── Light evaluation functions
│   ├── eval_point_light(light, n, v, pos, albedo, f0, roughness) → vec3
│   ├── eval_directional_light(light, n, v, pos, albedo, f0, roughness) → vec3  # With shadow
│   └── eval_spot_light(light, n, v, pos, albedo, f0, roughness) → vec3  # With cone atten
└── fs_main(input: FragmentInput) → @location(0) vec4<f32>
    ├── Read material from bindless table
    ├── Compute F0 = mix(0.04, albedo, metallic)
    ├── Loop over directional lights (shadowed)
    ├── Loop over point lights (distance attenuated)
    ├── Loop over spot lights (cone + distance attenuated)
    ├── Ambient = ambient_intensity * albedo * ao
    ├── Emissive = emissive.rgb * emissive.a
    └── Output = emissive + ambient + accumulated Lo
```

### PBR Vertex Shader (`shaders/pbr.vert.wgsl`)

```
pbr.vert.wgsl (64 lines)
├── CameraUniforms @group(0)     # view, projection, view_projection, camera_position
├── ModelUniforms @group(1)      # model, normal_matrix, material_index
├── VertexInput @location(0-3)   # position, normal, tangent, texcoord
├── VertexOutput @location(0-4)  # world_position, normal, tangent, texcoord, material_index
└── vs_main(input) → VertexOutput
    ├── world_pos = model * position
    ├── clip_position = view_projection * world_pos
    ├── world_normal = normalize(normal_matrix * normal)
    ├── world_tangent = normalize(normal_matrix * tangent)
    └── Pass material_index, texcoord
```

### Pipeline Infrastructure (`pipeline.rs`)

```
ShaderCache
├── modules: HashMap<[u8; 32], ShaderModule>   # SHA-256 keyed
├── source_hashes: HashMap<String, [u8; 32]>   # Path → hash
└── get_or_compile(device, source) → (ShaderModule, hash)

PipelineTable
├── pipelines: HashMap<u32, CachedPipeline>     # No LRU eviction
├── shader_cache: ShaderCache
├── insert(id, pipeline)
├── get(id) → Option<&CachedPipeline>
├── remove(id) → bool
└── compile_pipeline(device, id, source, vertex_entry, fragment_entry, layouts, format) → Result<u32>
```

### Renderer (`renderer.rs`)

```
Renderer
├── wgpu: Instance, Adapter, Device, Queue, Surface
├── render_pipeline: wgpu::RenderPipeline    # Triangle pipeline (NOT PBR)
├── vertex_buffer: wgpu::Buffer               # Triangle vertices (3)
├── uniform_buffer + bind_group               # Identity matrix uniform
└── new(window, width, height) → Self
    └── render() → renders a single colored triangle
```

## Missing for Functional PBR Pipeline

1. **PBR pipeline compilation** — Create a PBR render pipeline from `pbr.frag.wgsl` + `pbr.vert.wgsl` using `PipelineTable::compile_pipeline()`
2. **Mesh loading** — Load a mesh (e.g., glTF sphere) into vertex buffer + index buffer
3. **Bind group setup** — Wire CameraUniforms, MaterialTable, light buffers, shadow maps to PBR shader bind groups
4. **Frame graph pass** — Register the PBR pass in the frame graph IR
5. **Render loop** — Bind PBR pipeline, set bind groups, draw mesh

## Cross-References

- `crates/renderer-backend/shaders/pbr.frag.wgsl` — Complete PBR fragment shader
- `crates/renderer-backend/shaders/pbr.vert.wgsl` — Complete PBR vertex shader
- `crates/renderer-backend/src/pipeline.rs` — ShaderCache + PipelineTable (705 lines with tests)
- `crates/renderer-backend/src/renderer.rs` — wgpu Renderer (triangle only)
- `crates/renderer-backend/src/gpu_driven/material_table.rs` — Bindless MaterialTable
- `crates/renderer-backend/src/gpu_driven/material_table.wgsl` — WGSL companion
- `GAPSET_3_BRIDGE/GAP_3_SUMMARY.md` — Phase 6 PBR verification
