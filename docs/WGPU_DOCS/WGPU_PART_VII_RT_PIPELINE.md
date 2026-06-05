# WGPU_PART_VII_RT_PIPELINE.md — Ray Tracing Pipeline Deep Dive

> **TOC Reference**: Part VII, Chapters 13-14
> **Gate**: `ray_tracing_pipeline` + `shader_binding_table` (wgpu experimental)
> **Purpose**: Complete implementation spec — ready to execute when wgpu stabilizes
> **Generated**: 2026-05-26

---

## Preface: Why Document Gated Work

The RT Pipeline features are experimental in wgpu. Documenting them NOW serves three purposes:

1. **Architecture readiness**: TRINITY's abstractions must accommodate the full RT pipeline model from day one, even if Phase 1 uses inline ray queries.

2. **Upstream alignment**: Understanding wgpu's planned API surface lets us track upstream changes and participate in stabilization discussions.

3. **Zero-latency adoption**: When `ray_tracing_pipeline` stabilizes, TRINITY can adopt it immediately — no design work needed, just implementation.

---

# Chapter 13: Ray Tracing Pipelines

## 13.1 RT Pipeline Fundamentals

### 13.1.1 The Pipeline Model

Unlike raster/compute pipelines which have fixed shader stages, the RT pipeline is a **shader table dispatch** model:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Ray Tracing Pipeline                        │
│                                                                 │
│  ┌─────────────┐                                                │
│  │ Ray Gen     │  ← Entry point: generates initial rays         │
│  │ Shader      │                                                │
│  └──────┬──────┘                                                │
│         │ TraceRay()                                            │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │               Traversal (Hardware BVH)                   │   │
│  │                                                          │   │
│  │  For each ray-primitive intersection candidate:          │   │
│  │    ┌──────────────┐      ┌──────────────┐               │   │
│  │    │ Intersection │ ───▶ │   Any-Hit    │               │   │
│  │    │ Shader       │      │   Shader     │               │   │
│  │    │ (procedural) │      │ (alpha test) │               │   │
│  │    └──────────────┘      └──────────────┘               │   │
│  │                                                          │   │
│  │  On closest confirmed hit:    On miss (no hit):          │   │
│  │    ┌──────────────┐          ┌──────────────┐           │   │
│  │    │ Closest-Hit  │          │    Miss      │           │   │
│  │    │ Shader       │          │   Shader     │           │   │
│  │    └──────────────┘          └──────────────┘           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Optional recursive TraceRay() from hit/miss shaders            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 13.1.2 Pipeline vs Inline Ray Queries

| Aspect | Inline Ray Query | RT Pipeline |
|--------|------------------|-------------|
| **Entry point** | Compute shader | Ray generation shader |
| **Traversal control** | Manual loop (rayQueryProceed) | Automatic (TraceRay intrinsic) |
| **Hit processing** | Inline code | Separate hit shaders |
| **Material diversity** | Manual branching | SBT-driven dispatch |
| **Recursion** | Manual stack | Built-in (up to max depth) |
| **Performance** | Good for simple cases | Better for complex material graphs |
| **Flexibility** | High (full control) | Structured (shader table model) |

**When to use each:**
- **Ray Query**: Shadow rays, simple AO, uniform material handling
- **RT Pipeline**: Reflections with diverse materials, GI, path tracing

### 13.1.3 Shader Stages

| Stage | Attribute | Purpose | Invocation |
|-------|-----------|---------|------------|
| Ray Generation | `@raygeneration` | Generate rays, initiate tracing | Once per dispatch pixel |
| Intersection | `@intersection` | Custom intersection for procedural geometry | Per AABB candidate |
| Any-Hit | `@anyhit` | Accept/reject intersection (alpha test) | Per triangle candidate |
| Closest-Hit | `@closesthit` | Shade the closest intersection | Once per ray (if hit) |
| Miss | `@miss` | Handle rays that hit nothing | Once per ray (if miss) |
| Callable | `@callable` | Utility functions invoked from other stages | On demand |

### 13.1.4 Recursion Depth

RT pipelines support recursive tracing (e.g., reflection ray → shadow ray):

```
max_recursion_depth: u32

Typical values:
  1 = Primary ray only (no recursion)
  2 = Primary + one bounce (reflections OR shadows, not both)
  3 = Primary + reflection + shadow
  4 = Primary + reflection + secondary reflection + shadow
```

**TRINITY default**: `max_recursion_depth = 2` for real-time, `4` for high quality.

Higher recursion = more stack memory per ray = reduced parallelism.

---

## 13.2 Shader Stages Deep Dive

### 13.2.1 Ray Generation Shaders

**Purpose**: Entry point for ray tracing. Generates primary rays, calls TraceRay, writes results.

**WGSL structure** (speculative, based on Vulkan/DXR model):
```wgsl
@raygeneration
fn main() {
    // Built-in: launch dimensions and index
    let launch_id = raygeneration_launch_id();      // vec3<u32>
    let launch_size = raygeneration_launch_size();  // vec3<u32>
    
    // Compute pixel coordinates
    let pixel = vec2<f32>(launch_id.xy) + 0.5;
    let uv = pixel / vec2<f32>(launch_size.xy);
    
    // Generate ray from camera
    let ray_origin = camera.position;
    let ray_direction = compute_ray_direction(uv, camera);
    
    // Initialize payload (passed through hit/miss)
    var payload: RayPayload;
    payload.color = vec3(0.0);
    payload.depth = 0;
    
    // Trace ray
    traceRay(
        tlas,                           // acceleration structure
        RAY_FLAG_NONE,                  // ray flags
        0xFF,                           // cull mask
        0,                              // SBT offset
        1,                              // SBT stride
        0,                              // miss index
        ray_origin,                     // origin
        0.001,                          // t_min
        ray_direction,                  // direction
        10000.0,                        // t_max
        &payload                        // payload
    );
    
    // Write result
    textureStore(output_image, launch_id.xy, vec4(payload.color, 1.0));
}
```

**Key responsibilities:**
1. Determine ray origin and direction (from camera, G-buffer, or sampling)
2. Initialize ray payload struct
3. Call `traceRay()` intrinsic
4. Process returned payload
5. Write to output (texture or buffer)

### 13.2.2 Intersection Shaders

**Purpose**: Custom intersection testing for procedural geometry (spheres, SDFs, volumes).

**When invoked**: When a ray hits an AABB in the TLAS that contains procedural geometry.

**WGSL structure**:
```wgsl
@intersection
fn main() {
    // Built-in: ray parameters
    let ray_origin = intersection_ray_origin();
    let ray_direction = intersection_ray_direction();
    let t_min = intersection_ray_t_min();
    let t_max = intersection_ray_t_max();
    
    // Get primitive data from storage buffer (indexed by primitive ID)
    let prim_id = intersection_primitive_id();
    let sphere = spheres[prim_id];
    
    // Compute intersection
    let t = intersect_sphere(ray_origin, ray_direction, sphere);
    
    if (t >= t_min && t <= t_max) {
        // Report hit with custom attributes
        var attribs: SphereHitAttributes;
        attribs.normal = normalize((ray_origin + ray_direction * t) - sphere.center);
        
        reportIntersection(t, 0, attribs);  // t, hit_kind, attributes
    }
    // If no reportIntersection called, ray continues
}
```

**TRINITY use cases:**
- Sphere imposters
- SDF primitives
- Volumetric media boundaries
- Analytical curves/surfaces

### 13.2.3 Any-Hit Shaders

**Purpose**: Accept or reject intersection candidates before committing. Primary use: alpha testing.

**When invoked**: For each candidate intersection BEFORE it becomes the closest hit.

**WGSL structure**:
```wgsl
@anyhit
fn main(payload: ptr<ray_payload, RayPayload>) {
    // Get hit information
    let barycentrics = anyhit_barycentrics();  // vec2<f32>
    let prim_id = anyhit_primitive_id();
    let inst_id = anyhit_instance_id();
    let inst_custom = anyhit_instance_custom_index();
    
    // Fetch material from bindless table
    let material_idx = inst_custom;
    let material = materials[material_idx];
    
    // Compute UV from barycentrics
    let v0 = vertices[indices[prim_id * 3 + 0]];
    let v1 = vertices[indices[prim_id * 3 + 1]];
    let v2 = vertices[indices[prim_id * 3 + 2]];
    let uv = v0.uv * (1.0 - barycentrics.x - barycentrics.y) 
           + v1.uv * barycentrics.x 
           + v2.uv * barycentrics.y;
    
    // Sample alpha texture
    let alpha = textureSampleLevel(
        textures[material.alpha_texture_idx],
        linear_sampler,
        uv,
        0.0
    ).a;
    
    // Alpha test
    if (alpha < material.alpha_cutoff) {
        ignoreIntersection();  // Continue traversal
        return;
    }
    
    // Accept hit (default if no ignoreIntersection)
}
```

**Control flow:**
- `ignoreIntersection()` — reject this candidate, continue BVH traversal
- `terminateRay()` — accept this hit, stop traversal (shadow ray optimization)
- (no call) — accept candidate, continue finding closer hits

**Performance note**: Any-hit shaders are expensive. Mark OPAQUE geometry to skip them.

### 13.2.4 Closest-Hit Shaders

**Purpose**: Shade the closest intersection point. This is where PBR evaluation happens.

**When invoked**: Once per ray, after traversal completes, if a hit was found.

**WGSL structure**:
```wgsl
@closesthit
fn main(payload: ptr<ray_payload, RayPayload>, attribs: HitAttributes) {
    // Get hit information
    let barycentrics = attribs.barycentrics;
    let t = closesthit_ray_t();
    let prim_id = closesthit_primitive_id();
    let inst_id = closesthit_instance_id();
    let inst_custom = closesthit_instance_custom_index();
    let world_to_object = closesthit_world_to_object();
    let object_to_world = closesthit_object_to_world();
    
    // Compute world-space hit point
    let ray_origin = closesthit_ray_origin();
    let ray_direction = closesthit_ray_direction();
    let hit_point = ray_origin + ray_direction * t;
    
    // Interpolate vertex attributes
    let v0 = vertices[indices[prim_id * 3 + 0]];
    let v1 = vertices[indices[prim_id * 3 + 1]];
    let v2 = vertices[indices[prim_id * 3 + 2]];
    let w = 1.0 - barycentrics.x - barycentrics.y;
    
    let normal = normalize(
        v0.normal * w + v1.normal * barycentrics.x + v2.normal * barycentrics.y
    );
    let uv = v0.uv * w + v1.uv * barycentrics.x + v2.uv * barycentrics.y;
    
    // Transform normal to world space
    let world_normal = normalize((object_to_world * vec4(normal, 0.0)).xyz);
    
    // Fetch material
    let material_idx = inst_custom;
    let material = materials[material_idx];
    
    // Sample textures
    let albedo = textureSampleLevel(textures[material.albedo_idx], sampler, uv, 0.0).rgb;
    let roughness = textureSampleLevel(textures[material.roughness_idx], sampler, uv, 0.0).r;
    let metallic = textureSampleLevel(textures[material.metallic_idx], sampler, uv, 0.0).r;
    
    // PBR shading
    var color = vec3(0.0);
    
    // Direct lighting
    for (var i = 0u; i < light_count; i++) {
        let light = lights[i];
        let L = normalize(light.position - hit_point);
        let shadow = trace_shadow_ray(hit_point, L, light);
        color += evaluate_pbr(albedo, roughness, metallic, world_normal, -ray_direction, L) 
               * light.color * shadow;
    }
    
    // Indirect lighting (recursive trace)
    if (payload.depth < max_bounces) {
        let reflect_dir = reflect(ray_direction, world_normal);
        
        var reflect_payload: RayPayload;
        reflect_payload.depth = payload.depth + 1;
        
        traceRay(tlas, RAY_FLAG_NONE, 0xFF, 0, 1, 0,
                 hit_point + world_normal * 0.001, 0.001, reflect_dir, 10000.0,
                 &reflect_payload);
        
        let F = fresnel_schlick(max(dot(-ray_direction, world_normal), 0.0), 
                                mix(vec3(0.04), albedo, metallic));
        color += reflect_payload.color * F * (1.0 - roughness);
    }
    
    payload.color = color;
}
```

**Key patterns:**
1. Barycentric interpolation for smooth attributes
2. Material fetch via `instance_custom_index` → bindless table
3. PBR evaluation
4. Shadow ray tracing (optional recursion)
5. Reflection/refraction tracing (optional recursion)

### 13.2.5 Miss Shaders

**Purpose**: Handle rays that don't hit any geometry. Typically samples environment map.

**When invoked**: Once per ray, after traversal completes, if no hit was found.

**WGSL structure**:
```wgsl
@miss
fn main(payload: ptr<ray_payload, RayPayload>) {
    let ray_direction = miss_ray_direction();
    
    // Sample environment map
    let env_color = textureSampleLevel(
        environment_map,
        environment_sampler,
        ray_direction,
        0.0
    ).rgb;
    
    payload.color = env_color * environment_intensity;
}
```

**Multiple miss shaders**: Different miss indices can select different environment maps or behaviors:
- Index 0: Sky/environment
- Index 1: Shadow miss (return "not in shadow")
- Index 2: AO miss (return "no occlusion")

### 13.2.6 Callable Shaders

**Purpose**: Utility functions that can be invoked from any other RT shader stage.

**Use cases:**
- Shared BRDF evaluation
- Noise functions
- Complex material evaluation
- Dynamic dispatch to material-specific code

**WGSL structure**:
```wgsl
@callable
fn evaluate_glass(params: ptr<callable_data, GlassParams>) {
    let ior = params.ior;
    let roughness = params.roughness;
    
    // Complex glass BSDF evaluation
    params.result_color = ...;
    params.result_direction = ...;
}

// Called from closest-hit:
fn main(payload: ptr<ray_payload, RayPayload>, attribs: HitAttributes) {
    // ...
    var glass_params: GlassParams;
    glass_params.ior = material.ior;
    glass_params.roughness = material.roughness;
    
    executeCallable(CALLABLE_GLASS_IDX, &glass_params);
    
    payload.color = glass_params.result_color;
}
```

---

## 13.3 Hit Groups

### 13.3.1 Hit Group Concept

A **hit group** bundles shaders that handle a specific geometry/material type:

```
Hit Group = (Closest-Hit shader, Any-Hit shader*, Intersection shader*)
            * optional

Triangle Hit Group:    (closest-hit, any-hit?)
Procedural Hit Group:  (closest-hit, any-hit?, intersection)
```

### 13.3.2 Triangle Hit Groups

For standard triangle geometry:

| Hit Group | Closest-Hit | Any-Hit | Use Case |
|-----------|-------------|---------|----------|
| Opaque | `pbr.rchit` | (none) | Solid surfaces |
| Masked | `pbr.rchit` | `alpha_test.rahit` | Foliage, fences |
| Translucent | `glass.rchit` | `blend.rahit` | Glass, water |

### 13.3.3 Procedural Hit Groups

For AABB geometry with custom intersection:

| Hit Group | Intersection | Closest-Hit | Any-Hit | Use Case |
|-----------|--------------|-------------|---------|----------|
| Sphere | `sphere.rint` | `sphere.rchit` | (none) | Particle imposters |
| Volume | `volume.rint` | `volume.rchit` | (none) | Fog, clouds |
| SDF | `sdf.rint` | `sdf.rchit` | (none) | Analytical shapes |

### 13.3.4 Hit Group Indexing

The Shader Binding Table maps `(instance, geometry, ray_type)` → hit group:

```
hit_group_index = instance.sbt_offset 
                + geometry_index * sbt_stride 
                + ray_type
```

Where:
- `instance.sbt_offset` — per-instance offset in SBT (from TLAS instance descriptor)
- `geometry_index` — index of geometry within BLAS
- `sbt_stride` — number of ray types (e.g., 2 for primary + shadow)
- `ray_type` — 0 for primary rays, 1 for shadow rays, etc.

---

## 13.4 Shader Binding Table (SBT)

### 13.4.1 SBT Concept

The SBT is a GPU buffer containing **shader records**. Each record = shader handle + local data.

```
┌─────────────────────────────────────────────────────────────┐
│                    Shader Binding Table                      │
├─────────────────────────────────────────────────────────────┤
│  Ray Generation Records (one per launch configuration)      │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Record 0: raygen_shader_handle | local_data...     │    │
│  └────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│  Miss Records (one per miss shader)                         │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Record 0: miss_sky_handle | env_map_index          │    │
│  │ Record 1: miss_shadow_handle | (empty)             │    │
│  └────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│  Hit Group Records (one per hit group)                      │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Record 0: opaque_hit_group_handle | material_idx_0 │    │
│  │ Record 1: opaque_hit_group_handle | material_idx_1 │    │
│  │ Record 2: masked_hit_group_handle | material_idx_2 │    │
│  │ ...                                                 │    │
│  └────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│  Callable Records (one per callable shader)                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Record 0: callable_glass_handle | glass_params     │    │
│  │ Record 1: callable_subsurface_handle | sss_params  │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 13.4.2 SBT Layout in wgpu (Speculative)

Based on Vulkan/DXR patterns, wgpu's SBT will likely use:

```rust
struct ShaderBindingTableDescriptor {
    raygen: ShaderBindingTableRegion,
    miss: ShaderBindingTableRegion,
    hit: ShaderBindingTableRegion,
    callable: ShaderBindingTableRegion,
}

struct ShaderBindingTableRegion {
    buffer: wgpu::Buffer,
    offset: u64,
    stride: u64,
    size: u64,
}
```

### 13.4.3 Shader Record Structure

Each shader record has a header (shader handle) + optional local data:

```
┌─────────────────────────────────────────────────────┐
│ Shader Record (aligned to shader_record_alignment)   │
├─────────────────────────────────────────────────────┤
│ Shader Handle (32 bytes typically)                  │
│  - Opaque identifier for the shader group           │
├─────────────────────────────────────────────────────┤
│ Local Root Arguments (variable size)                │
│  - Material index                                   │
│  - Texture indices                                  │
│  - Custom parameters                                │
│  - (pad to alignment)                               │
└─────────────────────────────────────────────────────┘
```

### 13.4.4 TRINITY's SBT Builder

```rust
pub struct SbtBuilder {
    raygen_records: Vec<ShaderRecord>,
    miss_records: Vec<ShaderRecord>,
    hit_group_records: Vec<ShaderRecord>,
    callable_records: Vec<ShaderRecord>,
    
    handle_size: u32,
    record_alignment: u32,
}

impl SbtBuilder {
    pub fn new(pipeline: &RayTracingPipeline) -> Self {
        let properties = pipeline.get_shader_group_properties();
        Self {
            raygen_records: Vec::new(),
            miss_records: Vec::new(),
            hit_group_records: Vec::new(),
            callable_records: Vec::new(),
            handle_size: properties.shader_group_handle_size,
            record_alignment: properties.shader_group_base_alignment,
        }
    }
    
    pub fn add_raygen(&mut self, shader_index: u32, local_data: &[u8]) -> &mut Self {
        self.raygen_records.push(ShaderRecord {
            shader_index,
            local_data: local_data.to_vec(),
        });
        self
    }
    
    pub fn add_miss(&mut self, shader_index: u32, local_data: &[u8]) -> &mut Self {
        self.miss_records.push(ShaderRecord {
            shader_index,
            local_data: local_data.to_vec(),
        });
        self
    }
    
    pub fn add_hit_group(&mut self, shader_index: u32, local_data: &[u8]) -> &mut Self {
        self.hit_group_records.push(ShaderRecord {
            shader_index,
            local_data: local_data.to_vec(),
        });
        self
    }
    
    pub fn build(&self, device: &wgpu::Device, pipeline: &RayTracingPipeline) -> ShaderBindingTable {
        // Calculate sizes
        let raygen_stride = self.align_up(self.handle_size + self.max_local_size(&self.raygen_records));
        let miss_stride = self.align_up(self.handle_size + self.max_local_size(&self.miss_records));
        let hit_stride = self.align_up(self.handle_size + self.max_local_size(&self.hit_group_records));
        let callable_stride = self.align_up(self.handle_size + self.max_local_size(&self.callable_records));
        
        let raygen_size = raygen_stride * self.raygen_records.len() as u32;
        let miss_size = miss_stride * self.miss_records.len() as u32;
        let hit_size = hit_stride * self.hit_group_records.len() as u32;
        let callable_size = callable_stride * self.callable_records.len() as u32;
        
        let total_size = self.align_up(raygen_size) 
                       + self.align_up(miss_size)
                       + self.align_up(hit_size)
                       + self.align_up(callable_size);
        
        // Create buffer
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shader Binding Table"),
            size: total_size as u64,
            usage: wgpu::BufferUsages::SHADER_BINDING_TABLE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        
        // Write records
        {
            let mut data = buffer.slice(..).get_mapped_range_mut();
            let mut offset = 0;
            
            // Write raygen
            for record in &self.raygen_records {
                self.write_record(&mut data, offset, pipeline, record, raygen_stride);
                offset += raygen_stride as usize;
            }
            offset = self.align_up(offset as u32) as usize;
            
            // ... similar for miss, hit, callable
        }
        buffer.unmap();
        
        ShaderBindingTable {
            buffer,
            raygen: SbtRegion { offset: 0, stride: raygen_stride, size: raygen_size },
            miss: SbtRegion { offset: self.align_up(raygen_size), stride: miss_stride, size: miss_size },
            // ...
        }
    }
}
```

### 13.4.5 Material Domain → Hit Group Mapping

TRINITY organizes materials by domain:

| Material Domain | Hit Group Type | SBT Range |
|-----------------|----------------|-----------|
| `OPAQUE` | Triangle (no any-hit) | 0-N |
| `MASKED` | Triangle (with alpha any-hit) | N-M |
| `TRANSLUCENT` | Triangle (with blend any-hit) | M-P |
| `VOLUME` | Procedural (volume intersection) | P-Q |
| `SUBSURFACE` | Triangle (SSS closest-hit) | Q-R |

Instance `sbt_offset` is set based on material domain:
```rust
let sbt_offset = match material.domain {
    MaterialDomain::Opaque => 0,
    MaterialDomain::Masked => num_opaque_hit_groups,
    MaterialDomain::Translucent => num_opaque_hit_groups + num_masked_hit_groups,
    // ...
};
```

---

## 13.5 RT Pipeline Creation

### 13.5.1 wgpu RT Pipeline (Speculative API)

```rust
let pipeline = device.create_ray_tracing_pipeline(&wgpu::RayTracingPipelineDescriptor {
    label: Some("RT Reflections"),
    layout: Some(&pipeline_layout),
    
    // Shader stages
    stages: &[
        wgpu::RayTracingShaderStage {
            module: &raygen_module,
            entry_point: "main",
            stage: wgpu::RayTracingStage::RayGeneration,
        },
        wgpu::RayTracingShaderStage {
            module: &miss_module,
            entry_point: "main",
            stage: wgpu::RayTracingStage::Miss,
        },
        wgpu::RayTracingShaderStage {
            module: &closesthit_module,
            entry_point: "main",
            stage: wgpu::RayTracingStage::ClosestHit,
        },
        wgpu::RayTracingShaderStage {
            module: &anyhit_module,
            entry_point: "main",
            stage: wgpu::RayTracingStage::AnyHit,
        },
    ],
    
    // Shader groups (define how stages combine into hit groups)
    groups: &[
        // Group 0: Ray generation
        wgpu::RayTracingShaderGroup::General { 
            general: 0,  // Index into stages[]
        },
        // Group 1: Miss
        wgpu::RayTracingShaderGroup::General { 
            general: 1,
        },
        // Group 2: Opaque hit group (closest-hit only)
        wgpu::RayTracingShaderGroup::TrianglesHitGroup {
            closest_hit: Some(2),
            any_hit: None,
        },
        // Group 3: Masked hit group (closest-hit + any-hit)
        wgpu::RayTracingShaderGroup::TrianglesHitGroup {
            closest_hit: Some(2),
            any_hit: Some(3),
        },
    ],
    
    max_pipeline_ray_recursion_depth: 2,
    
    // Optional: pipeline library for faster creation
    library: None,
});
```

### 13.5.2 Pipeline Layout

RT pipelines share bind group layouts with compute:

```rust
let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
    label: Some("RT Pipeline Layout"),
    bind_group_layouts: &[
        &scene_bind_group_layout,     // Group 0: TLAS, camera, lights
        &material_bind_group_layout,  // Group 1: Materials, textures
        &output_bind_group_layout,    // Group 2: Output images
    ],
    push_constant_ranges: &[
        wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::all(),
            range: 0..16,  // Frame data
        },
    ],
});
```

### 13.5.3 TRINITY's RT Pipeline Cache

```rust
pub struct RtPipelineCache {
    pipelines: HashMap<RtPipelineKey, Arc<wgpu::RayTracingPipeline>>,
    sbt_cache: HashMap<SbtKey, Arc<ShaderBindingTable>>,
}

#[derive(Hash, Eq, PartialEq)]
struct RtPipelineKey {
    effect_type: RtEffectType,  // Shadows, Reflections, GI, PathTrace
    material_domains: MaterialDomainMask,
    max_recursion: u32,
    quality_level: QualityLevel,
}

impl RtPipelineCache {
    pub fn get_or_create(
        &mut self,
        device: &wgpu::Device,
        key: &RtPipelineKey,
    ) -> Arc<wgpu::RayTracingPipeline> {
        self.pipelines.entry(key.clone()).or_insert_with(|| {
            Arc::new(self.create_pipeline(device, key))
        }).clone()
    }
}
```

---

## 13.6 Ray Tracing Dispatch

### 13.6.1 TraceRay Intrinsic

The core operation in RT shaders:

```wgsl
fn traceRay(
    tlas: acceleration_structure,
    ray_flags: u32,
    cull_mask: u32,
    sbt_record_offset: u32,
    sbt_record_stride: u32,
    miss_index: u32,
    origin: vec3<f32>,
    t_min: f32,
    direction: vec3<f32>,
    t_max: f32,
    payload: ptr<ray_payload, T>,
)
```

### 13.6.2 Dispatch Dimensions

RT dispatch is similar to compute, but conceptually maps to screen pixels:

```rust
encoder.begin_ray_tracing_pass(&wgpu::RayTracingPassDescriptor {
    label: Some("RT Reflections"),
});

encoder.set_pipeline(&rt_pipeline);
encoder.set_bind_group(0, &scene_bind_group, &[]);
encoder.set_bind_group(1, &material_bind_group, &[]);
encoder.set_bind_group(2, &output_bind_group, &[]);

// Dispatch with SBT regions
encoder.trace_rays(
    &sbt.raygen,   // Ray generation region
    &sbt.miss,     // Miss region
    &sbt.hit,      // Hit group region
    &sbt.callable, // Callable region
    width,         // Dispatch width (typically screen width or half)
    height,        // Dispatch height
    1,             // Dispatch depth (typically 1)
);

encoder.end_ray_tracing_pass();
```

### 13.6.3 Payload Passing

Payloads are per-ray data passed between shader stages:

```wgsl
struct RayPayload {
    color: vec3<f32>,
    depth: u32,
    hit_t: f32,
    // ... custom fields
}
```

- Ray gen initializes payload → Closest-hit/miss write results → Ray gen reads back
- Recursive traces can use nested payloads or reuse the same

### 13.6.4 Attribute Passing

Attributes are per-hit data from intersection/hardware:

```wgsl
struct HitAttributes {
    barycentrics: vec2<f32>,  // Built-in for triangles
    // Custom fields for procedural geometry
}
```

---

## 13.7 RT Pipeline Patterns

### 13.7.1 Primary Ray Casting

Basic pattern for primary visibility:

```
RayGen:
  for each pixel:
    generate camera ray
    trace(ray) → payload
    write payload.color to output
```

### 13.7.2 Shadow Rays with SBT

Optimized shadow rays using dedicated miss shader:

```
RayGen:
  generate primary ray
  trace(ray, miss_index=0) → payload
  
ClosestHit:
  for each light:
    trace(shadow_ray, 
          sbt_offset=SHADOW_OFFSET, 
          miss_index=1)  // Shadow miss returns 1.0
    accumulate lighting * shadow

MissShadow:
  payload.shadow = 1.0  // No occlusion

ClosestHitShadow:
  (empty — just existing terminates ray with occlusion)
```

### 13.7.3 Reflections

Multi-bounce reflections:

```
RayGen:
  trace primary → payload
  output = payload.color
  
ClosestHit:
  shade direct lighting
  if (depth < max_bounce && material.reflective):
    compute reflect direction
    trace(reflect_ray) → reflect_payload
    color += reflect_payload.color * fresnel
  payload.color = color
```

### 13.7.4 Global Illumination

Single-bounce diffuse GI:

```
RayGen:
  for each G-buffer pixel:
    sample hemisphere direction (cosine weighted)
    trace(gi_ray) → payload
    output indirect = payload.color * albedo / PI
    
ClosestHit:
  evaluate direct lighting only (no recursion)
  payload.color = direct_radiance
  
Miss:
  payload.color = environment
```

### 13.7.5 Path Tracing

Multi-bounce with Russian roulette:

```
RayGen:
  generate camera ray
  throughput = 1.0
  radiance = 0.0
  
  for bounce in 0..max_bounces:
    trace(ray) → payload
    if (miss): radiance += throughput * env; break
    
    // NEE (Next Event Estimation)
    radiance += throughput * payload.direct
    
    // Continue path
    throughput *= payload.bsdf / pdf
    ray = payload.next_ray
    
    // Russian roulette
    if (bounce > 2):
      p_continue = max(throughput)
      if (random() > p_continue): break
      throughput /= p_continue
  
  output = radiance
```

---

# Chapter 14: RT Advanced Features

## 14.1 Opacity Micromaps (OMM)

### 14.1.1 The Problem OMM Solves

Alpha-tested geometry (foliage, fences) requires any-hit shader invocation for EVERY ray-triangle intersection. This is expensive.

OMM encodes per-micropolygon opacity into the acceleration structure, allowing the hardware to:
- Skip any-hit for fully opaque regions
- Skip traversal entirely for fully transparent regions
- Only invoke any-hit for mixed regions

### 14.1.2 OMM Data Structure

```
OMM = per-triangle micropolygon grid
  - Resolution: 1, 2, 4, 8, 16 subdivisions per edge
  - States: Transparent (0), Opaque (1), Unknown (2)
  
Triangle with alpha texture:
┌─────────────────────┐
│ O O O O O O O O O O │  O = Opaque
│ O O O O O O O O ? ? │  T = Transparent  
│ O O O O O O ? ? T T │  ? = Unknown (needs any-hit)
│ O O O O ? ? T T T T │
│ O O ? ? T T T T T T │
│ ? ? T T T T T T T T │
└─────────────────────┘
```

### 14.1.3 OMM Building Pipeline

```rust
// 1. For each alpha-tested material, generate OMM data
let omm_data = generate_omm_for_mesh(
    mesh,
    alpha_texture,
    alpha_threshold,
    omm_resolution,  // 4 or 8 typical
);

// 2. Attach OMM to BLAS build
let blas_desc = wgpu::BlasDescriptor {
    geometries: &[geometry],
    opacity_micromap: Some(&omm_data),
    // ...
};
```

### 14.1.4 wgpu OMM Status

As of 2026-05: Not yet implemented. On wgpu roadmap post-RT-pipeline stabilization.

**TRINITY preparation**: Design BLAS builder with OMM slot; implement CPU-side OMM generation; enable when wgpu adds support.

## 14.2 Displacement Micromaps (DMM)

### 14.2.1 DMM Concept

DMM encodes micro-geometry displacement into the acceleration structure:
- Base mesh is coarse
- DMM adds detail for intersection without increasing triangle count
- BVH is tighter than base mesh bounds

### 14.2.2 Use Cases

- Terrain heightmaps
- Tiled surface detail
- Micro-geometry without tessellation

### 14.2.3 wgpu DMM Status

Not on near-term roadmap. Requires NVIDIA RTX 40-series or equivalent.

## 14.3 Shader Execution Reordering (SER)

### 14.3.1 The Coherence Problem

In path tracing, rays diverge wildly:
- One ray hits metal → simple reflection
- Adjacent ray hits glass → complex refraction
- Another ray hits subsurface skin → SSS evaluation

This causes warp/wave divergence: some threads wait while others do heavy work.

### 14.3.2 SER Solution

SER allows the hardware to **reorder shader execution** to improve coherence:

```
Before SER:
  Thread 0: Metal    (simple)   ████░░░░░░
  Thread 1: Glass    (complex)  ██████████
  Thread 2: Skin     (very complex) ████████████████
  Thread 3: Metal    (simple)   ████░░░░░░
  ↓ All threads wait for Thread 2

After SER:
  Reordered so similar materials execute together
  Thread 0,3: Metal batch → ████
  Thread 1: Glass batch   → ██████████
  Thread 2: Skin batch    → ████████████████
  ↓ Better utilization
```

### 14.3.3 WGSL SER Hints (Speculative)

```wgsl
@closesthit
fn main(payload: ptr<ray_payload, RayPayload>) {
    // Hint for SER: material type
    reorderThread(material.domain);
    
    // ... shading code
}
```

### 14.3.4 wgpu SER Status

NVIDIA-specific (Shader Execution Reordering). Not in wgpu roadmap.

**TRINITY approach**: Design material system with coherence in mind; benefit from SER when available.

## 14.4 Motion Blur

### 14.4.1 Motion BLAS/TLAS

For motion blur, acceleration structures store transforms at multiple time samples:

```
Motion TLAS Instance:
  - transform_t0: mat4  (start of frame)
  - transform_t1: mat4  (end of frame)
  - motion_type: LINEAR | SRT

Motion BLAS:
  - vertex_positions_t0
  - vertex_positions_t1
```

### 14.4.2 Tracing with Time

```wgsl
traceRayMotion(
    tlas,
    ray_flags,
    cull_mask,
    sbt_offset, sbt_stride, miss_index,
    origin, t_min, direction, t_max,
    time,  // [0, 1] within frame
    payload
);
```

### 14.4.3 wgpu Motion Blur Status

Not on near-term roadmap. Requires significant AS changes.

---

# Implementation Roadmap for TRINITY

## Immediate (Pre-wgpu Stabilization)

1. **Design SBT abstraction** — Build `SbtBuilder` API now, mock wgpu calls
2. **Material domain system** — Categorize materials for efficient SBT layout
3. **Hit group registry** — Define hit groups per material domain
4. **Shader templates** — Write WGSL scaffolds for all shader stages
5. **Pipeline key design** — Define cache keys for RT pipeline variants

## On wgpu Stabilization

1. **Implement SBT buffer management**
2. **Implement RT pipeline creation**
3. **Wire up TraceRay dispatch**
4. **Integration tests with simple scenes**
5. **Performance tuning (recursion depth, SBT layout)**

## Post-Stabilization

1. **OMM integration** (when available)
2. **Multiple ray types** (primary, shadow, GI, AO)
3. **Hybrid rendering** (raster + RT)
4. **Path tracing mode**

---

# Cross-References

| Section | Related GAPSET Task |
|---------|---------------------|
| §13.5 | T-RT-P2.1 (RT pipeline creation) |
| §13.4 | T-RT-P2.2 (SBT builder) |
| §13.2.4 | T-RT-P2.3 (RT reflection shaders) |
| §13.7.4 | T-RT-P2.4 (RT GI shaders) |
| §13.6 | T-RT-P2.5, T-RT-P2.6 (dispatch) |
| §14.1 | T-RT-P3.5 (OMM timeline research) |

---

*End of WGPU_PART_VII_RT_PIPELINE.md*
