# PHASE 5: RAY_TRACING - Architecture

**Scope:** Acceleration structures, ray queries, ray tracing pipelines
**Duration:** 4-6 weeks
**Dependencies:** Phase 4 (SYNCHRONIZATION)
**Produces:** Complete ray tracing capability (when wgpu stabilizes)

---

## Overview

Phase 5 implements wgpu's ray tracing features: acceleration structure building, inline ray queries for compute/fragment shaders, and the full ray tracing pipeline. Note that RT pipeline is experimental in wgpu.

### Covered Content (from MASTER.md Part VII)

- Chapter 11: Acceleration Structures
  - 11.1 Fundamentals (BVH, BLAS/TLAS hierarchy)
  - 11.2 BLAS construction (geometry types, flags, scratch buffers)
  - 11.3 BLAS compaction (memory savings)
  - 11.4 BLAS update (refit vs rebuild)
  - 11.5 TLAS construction (instance descriptor, per-frame rebuild)
  - 11.6 AS memory management

- Chapter 12: Ray Queries (Inline)
  - 12.1 Fundamentals (ray_query feature, use cases)
  - 12.2 WGSL API (RayQuery type, proceed/commit)
  - 12.3 Ray flags
  - 12.4 Patterns (shadow, closest hit, any-hit)

- Chapter 13: Ray Tracing Pipelines (Experimental)
  - 13.1 Fundamentals (stages, recursion)
  - 13.2 Shader stages (6 types)
  - 13.3 Hit groups
  - 13.4 Shader Binding Table (SBT)
  - 13.5 Pipeline creation
  - 13.6 Dispatch
  - 13.7 Patterns (primary rays, shadows, GI)

- Chapter 14: RT Advanced Features (Future)
  - 14.1 Opacity Micromaps (OMM)
  - 14.2 Displacement Micromaps (DMM)
  - 14.3 Shader Execution Reordering (SER)
  - 14.4 Motion blur

---

## Architectural Decisions

### ADR-017: Two-Level Acceleration Structure

**Context:** Ray tracing requires efficient spatial data structures.

**Decision:** Implement standard two-level hierarchy:
- BLAS: Per-mesh geometry, built once, updated for deformation
- TLAS: Per-instance references, rebuilt every frame

**Rationale:** Industry standard for real-time RT.

**Consequences:**
- BLAS reused across instances
- TLAS rebuild cost per frame
- Transform updates cheap (TLAS only)

---

### ADR-018: Ray Query vs RT Pipeline

**Context:** Two approaches to ray tracing in wgpu.

**Decision:** Support both:
- Ray Query: Stable, inline in compute/fragment, preferred for shadows/AO
- RT Pipeline: Experimental, full shader stages, for advanced effects

**Rationale:** Ray Query is production-ready; RT Pipeline prepared for future.

**Consequences:**
- Ray Query code shipped now
- RT Pipeline code behind feature flag
- Effect selection based on availability

---

### ADR-019: BLAS Update Policy

**Context:** Animated geometry needs BLAS updates.

**Decision:** Implement dynamic policy:
- Static meshes: Build once, no updates
- Skinned meshes: Refit up to N frames, then rebuild
- Highly dynamic: Rebuild each frame

**Rationale:** Balances quality vs performance.

**Consequences:**
- Quality tracking for refit degradation
- Automatic rebuild trigger
- Per-mesh policy configuration

---

### ADR-020: SBT Layout Strategy

**Context:** Shader Binding Table layout affects flexibility and performance.

**Decision:** Implement structured SBT builder:
- Ray gen records at offset 0
- Miss records at fixed offset
- Hit group records indexed by (instance_index * sbt_offset + geometry_index)

**Rationale:** Matches DXR/Vulkan conventions.

**Consequences:**
- Predictable SBT layout
- Hit group indexing formula documented
- Multi-material support

---

## Component Breakdown

### 1. Acceleration Structures

```
TrinityASBuilder
├── device: Arc<Device>
├── scratch_pool: ScratchBufferPool
├── blas_cache: HashMap<MeshId, Arc<BLAS>>
└── tlas: Option<TLAS>
```

**BLAS:**
- geometry: Vec<BlasGeometry>
- flags: BlasBuildFlags
- buffer: wgpu::Buffer (AS data)
- scratch: ScratchAllocation
- compacted: bool
- quality: f32 (1.0 = perfect, degrades with refit)

**BlasGeometry:**
- GeometryTriangles { vertex_buffer, index_buffer, transform }
- GeometryAABBs { aabb_buffer }

**BlasBuildFlags:**
- PREFER_FAST_TRACE
- PREFER_FAST_BUILD
- ALLOW_UPDATE
- ALLOW_COMPACTION
- LOW_MEMORY

**TLAS:**
- instances: Vec<TlasInstance>
- buffer: wgpu::Buffer
- instance_buffer: wgpu::Buffer

**TlasInstance:**
- blas: Arc<BLAS>
- transform: [[f32; 4]; 3]
- instance_id: u32
- mask: u8
- sbt_offset: u32
- flags: InstanceFlags

### 2. BLAS Operations

```
BLASManager
├── builder: TrinityASBuilder
├── pending_builds: Vec<BlasBuildRequest>
├── pending_compactions: Vec<BlasCompactRequest>
└── dynamic_policy: DynamicBlasPolicy
```

**Operations:**
- `build_blas()` - Initial construction
- `compact_blas()` - Post-build compaction
- `update_blas()` - Refit existing
- `rebuild_blas()` - Full rebuild

**DynamicBlasPolicy:**
- max_refit_frames: u32 (default: 30)
- quality_threshold: f32 (default: 0.7)
- force_rebuild_on_threshold: bool

### 3. TLAS Operations

```
TLASManager
├── builder: TrinityASBuilder
├── instance_staging: Buffer
└── build_scratch: Buffer
```

**Operations:**
- `build_tlas()` - From instance list
- `update_instances()` - Update transforms only
- `cull_instances()` - Pre-build visibility cull

### 4. Ray Query System

```
TrinityRayQuery
├── as_manager: TLASManager
├── shadow_queries: Vec<ShadowQuery>
└── reflection_queries: Vec<ReflectionQuery>
```

**Ray Query WGSL:**
```wgsl
var<private> rq: ray_query;

fn trace_shadow(origin: vec3<f32>, direction: vec3<f32>, max_t: f32) -> bool {
    rayQueryInitialize(&rq, tlas, 
        RAY_FLAG_TERMINATE_ON_FIRST_HIT | RAY_FLAG_FORCE_OPAQUE,
        0xFFu, origin, 0.001, direction, max_t);
    
    while (rayQueryProceed(&rq)) {}
    
    return rayQueryGetCommittedIntersectionType(&rq) != RAY_QUERY_COMMITTED_INTERSECTION_NONE;
}
```

**Use Cases:**
- Shadow rays (early termination)
- Ambient occlusion (hemisphere sampling)
- Reflections (single bounce)
- Contact hardening shadows

### 5. RT Pipeline (Experimental)

```
TrinityRTPipeline
├── pipeline: wgpu::RayTracingPipeline
├── sbt: ShaderBindingTable
├── recursion_depth: u32
└── payload_size: u32
```

**ShaderBindingTable:**
- ray_gen_records: Vec<SbtRecord>
- miss_records: Vec<SbtRecord>
- hit_group_records: Vec<SbtRecord>
- callable_records: Vec<SbtRecord>
- buffer: wgpu::Buffer

**SbtRecord:**
- shader_id: ShaderId
- local_data: Vec<u8> (up to 32 bytes)

**Shader Stages:**
| Stage | Attribute | Purpose |
|-------|-----------|---------|
| Ray Gen | @raygeneration | Camera rays, dispatch entry |
| Intersection | @intersection | Procedural geometry |
| Any-Hit | @anyhit | Alpha testing, transparency |
| Closest-Hit | @closesthit | Shading, recursion |
| Miss | @miss | Environment, sky |
| Callable | @callable | Utility functions |

### 6. RT Effects Library

```
TrinityRTEffects
├── primary_rays: RTPipeline
├── shadow_rays: RayQueryPipeline
├── reflection_rays: RayQueryPipeline
├── ao_rays: ComputePipeline (ray query)
├── gi_single_bounce: RTPipeline
└── path_tracer: RTPipeline
```

**Effect Selection:**
- Full tier: RT pipeline effects
- Ray query: Ray query effects
- Lower tiers: Fallback (screen-space)

---

## Module Structure

```
crates/renderer-backend/src/ray_tracing/
├── mod.rs              # Module exports
├── acceleration/
│   ├── mod.rs          # AS exports
│   ├── blas.rs         # BLAS builder
│   ├── tlas.rs         # TLAS builder
│   ├── scratch.rs      # Scratch buffer pool
│   ├── compaction.rs   # BLAS compaction
│   └── policy.rs       # Dynamic update policy
│
├── ray_query/
│   ├── mod.rs          # Ray query exports
│   ├── shadow.rs       # Shadow ray queries
│   ├── ao.rs           # AO ray queries
│   └── reflection.rs   # Reflection queries
│
├── pipeline/
│   ├── mod.rs          # RT pipeline exports
│   ├── sbt.rs          # SBT builder
│   ├── hit_groups.rs   # Hit group management
│   └── stages.rs       # Shader stage handling
│
├── effects/
│   ├── mod.rs          # Effect exports
│   ├── shadows.rs      # RT shadows
│   ├── reflections.rs  # RT reflections
│   ├── gi.rs           # Global illumination
│   └── path_tracer.rs  # Path tracing
│
└── shaders/
    ├── ray_gen.wgsl
    ├── closest_hit.wgsl
    ├── miss.wgsl
    ├── shadow_ray_query.wgsl
    └── ao_ray_query.wgsl
```

---

## Testing Strategy

### Unit Tests

1. **BLAS construction** - Geometry types, flags
2. **TLAS construction** - Instance transforms
3. **SBT layout** - Record offsets, indexing
4. **Ray query** - Intersection results
5. **Dynamic policy** - Quality degradation

### Integration Tests

1. **AS build** - Full BLAS/TLAS pipeline
2. **Shadow tracing** - Binary visibility
3. **Reflection** - Hit results
4. **Compaction** - Memory savings
5. **Refit** - Quality tracking

### Visual Tests

1. **RT shadows** - Compare to rasterized
2. **RT reflections** - Mirror surfaces
3. **RT AO** - Hemisphere sampling
4. **Path tracer** - Reference renderer

---

## Performance Considerations

1. **BLAS Sharing** - Reuse across instances
2. **TLAS Rebuild** - Per-frame cost acceptable
3. **Compaction** - 40-60% memory savings typical
4. **Ray Coherence** - Sort rays by direction
5. **Incoherent Rays** - Limit secondary bounces

---

## Dependencies

### External Crates

- `wgpu` - Core GPU abstraction (with ray tracing features)
- `glam` - Transform matrices

### Internal Dependencies

- Phase 1: TrinityDevice
- Phase 2: TrinityBufferSystem
- Phase 3: TrinityComputePipeline
- Phase 4: TrinityCommandEncoder

---

## Deliverables Checklist

- [ ] BLAS builder with geometry types
- [ ] BLAS compaction
- [ ] BLAS update/refit
- [ ] TLAS builder
- [ ] Scratch buffer pool
- [ ] Dynamic update policy
- [ ] Ray query shadow tracing
- [ ] Ray query AO
- [ ] Ray query reflections
- [ ] RT pipeline (experimental, feature-gated)
- [ ] SBT builder
- [ ] RT effects library
- [ ] Unit tests
- [ ] Integration tests
- [ ] Visual tests
- [ ] Documentation

---

*End of PHASE_5_RAY_TRACING_ARCH.md*
