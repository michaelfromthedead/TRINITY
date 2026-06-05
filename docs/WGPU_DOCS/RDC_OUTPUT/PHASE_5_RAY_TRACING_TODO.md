# PHASE 5: RAY_TRACING - Task List

**Phase:** 5 - RAY_TRACING
**Estimated Duration:** 4-6 weeks
**Task ID Prefix:** T-WGPU-P5

---

## Task Summary

| ID | Task | Est. Hours | Status |
|----|------|------------|--------|
| T-WGPU-P5.1.1 | BLAS geometry triangles | 6 | - |
| T-WGPU-P5.1.2 | BLAS geometry AABBs | 4 | - |
| T-WGPU-P5.1.3 | BLAS build flags | 3 | - |
| T-WGPU-P5.1.4 | Scratch buffer pool | 6 | - |
| T-WGPU-P5.1.5 | BLAS construction | 8 | - |
| T-WGPU-P5.2.1 | BLAS compaction query | 4 | - |
| T-WGPU-P5.2.2 | BLAS compaction copy | 4 | - |
| T-WGPU-P5.2.3 | Compaction decision logic | 3 | - |
| T-WGPU-P5.3.1 | BLAS update (refit) | 6 | - |
| T-WGPU-P5.3.2 | Quality tracking | 4 | - |
| T-WGPU-P5.3.3 | Dynamic update policy | 4 | - |
| T-WGPU-P5.4.1 | TLAS instance descriptor | 4 | - |
| T-WGPU-P5.4.2 | TLAS construction | 6 | - |
| T-WGPU-P5.4.3 | TLAS per-frame rebuild | 4 | - |
| T-WGPU-P5.4.4 | Instance culling integration | 4 | - |
| T-WGPU-P5.5.1 | AS memory manager | 6 | - |
| T-WGPU-P5.5.2 | AS memory budget | 4 | - |
| T-WGPU-P5.6.1 | Ray query fundamentals | 4 | - |
| T-WGPU-P5.6.2 | Ray query WGSL API | 6 | - |
| T-WGPU-P5.6.3 | Ray flags | 3 | - |
| T-WGPU-P5.6.4 | Shadow ray pattern | 6 | - |
| T-WGPU-P5.6.5 | AO ray pattern | 6 | - |
| T-WGPU-P5.6.6 | Reflection ray pattern | 6 | - |
| T-WGPU-P5.7.1 | RT pipeline descriptor | 6 | - |
| T-WGPU-P5.7.2 | Ray gen shader stage | 4 | - |
| T-WGPU-P5.7.3 | Intersection shader stage | 4 | - |
| T-WGPU-P5.7.4 | Any-hit shader stage | 4 | - |
| T-WGPU-P5.7.5 | Closest-hit shader stage | 4 | - |
| T-WGPU-P5.7.6 | Miss shader stage | 3 | - |
| T-WGPU-P5.7.7 | Callable shader stage | 3 | - |
| T-WGPU-P5.8.1 | Hit group definition | 4 | - |
| T-WGPU-P5.8.2 | SBT layout | 6 | - |
| T-WGPU-P5.8.3 | SBT builder | 8 | - |
| T-WGPU-P5.8.4 | Hit group indexing | 4 | - |
| T-WGPU-P5.9.1 | RT dispatch | 4 | - |
| T-WGPU-P5.9.2 | Primary ray effect | 6 | - |
| T-WGPU-P5.9.3 | RT shadows effect | 6 | - |
| T-WGPU-P5.9.4 | RT reflections effect | 6 | - |
| T-WGPU-P5.9.5 | Single bounce GI | 8 | - |
| T-WGPU-P5.10.1 | Unit tests | 8 | - |
| T-WGPU-P5.10.2 | Integration tests | 8 | - |
| T-WGPU-P5.10.3 | Visual tests | 6 | - |

**Total Estimated Hours:** 208 hours

---

## Detailed Tasks

### T-WGPU-P5.1.1 - BLAS Geometry Triangles

**Description:** Implement triangle geometry for BLAS.

**Prerequisites:** Phase 4 complete

**Deliverable:** BlasTriangleGeometry struct

**Acceptance Criteria:**
- [ ] Vertex buffer reference
- [ ] Index buffer reference (optional)
- [ ] Vertex format specification
- [ ] Transform buffer (optional)
- [ ] Geometry flags (OPAQUE, NO_DUPLICATE_ANYHIT)

**Estimate:** 6 hours

---

### T-WGPU-P5.1.2 - BLAS Geometry AABBs

**Description:** Implement AABB geometry for procedural BLAS.

**Prerequisites:** T-WGPU-P5.1.1

**Deliverable:** BlasAABBGeometry struct

**Acceptance Criteria:**
- [ ] AABB buffer (min, max per primitive)
- [ ] Stride specification
- [ ] Primitive count
- [ ] Use case documentation (particles, SDFs)

**Estimate:** 4 hours

---

### T-WGPU-P5.1.3 - BLAS Build Flags

**Description:** Implement BLAS build flag configuration.

**Prerequisites:** T-WGPU-P5.1.1

**Deliverable:** BlasBuildFlags enum/bitflags

**Acceptance Criteria:**
- [ ] PREFER_FAST_TRACE (default for static)
- [ ] PREFER_FAST_BUILD (for dynamic)
- [ ] ALLOW_UPDATE (enables refit)
- [ ] ALLOW_COMPACTION
- [ ] LOW_MEMORY
- [ ] Flag combination documentation

**Estimate:** 3 hours

---

### T-WGPU-P5.1.4 - Scratch Buffer Pool

**Description:** Implement scratch buffer pooling for AS builds.

**Prerequisites:** Phase 2 buffers

**Deliverable:** ScratchBufferPool struct

**Acceptance Criteria:**
- [ ] Size-class pooling
- [ ] Acquire/release pattern
- [ ] Alignment requirements
- [ ] Memory tracking

**Estimate:** 6 hours

---

### T-WGPU-P5.1.5 - BLAS Construction

**Description:** Implement BLAS build command.

**Prerequisites:** T-WGPU-P5.1.1 through T-WGPU-P5.1.4

**Deliverable:** `build_blas()` method

**Acceptance Criteria:**
- [ ] Feature check (acceleration_structure)
- [ ] Build sizes query
- [ ] Scratch buffer allocation
- [ ] Build command encoding
- [ ] Result buffer creation

**Estimate:** 8 hours

---

### T-WGPU-P5.2.1 - BLAS Compaction Query

**Description:** Implement compacted size query.

**Prerequisites:** T-WGPU-P5.1.5

**Deliverable:** Compaction size query

**Acceptance Criteria:**
- [ ] Query set for compacted size
- [ ] Async result readback
- [ ] Size comparison logging

**Estimate:** 4 hours

---

### T-WGPU-P5.2.2 - BLAS Compaction Copy

**Description:** Implement compaction copy command.

**Prerequisites:** T-WGPU-P5.2.1

**Deliverable:** `compact_blas()` method

**Acceptance Criteria:**
- [ ] New buffer allocation
- [ ] Copy AS command
- [ ] Old buffer deferred destroy
- [ ] Memory savings tracking

**Estimate:** 4 hours

---

### T-WGPU-P5.2.3 - Compaction Decision Logic

**Description:** Implement when to compact vs skip.

**Prerequisites:** T-WGPU-P5.2.2

**Deliverable:** Compaction policy

**Acceptance Criteria:**
- [ ] Threshold (e.g., >10% savings)
- [ ] Async compaction (background)
- [ ] Skip for small BLAS
- [ ] Configuration options

**Estimate:** 3 hours

---

### T-WGPU-P5.3.1 - BLAS Update (Refit)

**Description:** Implement BLAS refit for animated geometry.

**Prerequisites:** T-WGPU-P5.1.5

**Deliverable:** `update_blas()` method

**Acceptance Criteria:**
- [ ] ALLOW_UPDATE flag required
- [ ] In-place update command
- [ ] Scratch buffer for update
- [ ] Quality degradation warning

**Estimate:** 6 hours

---

### T-WGPU-P5.3.2 - Quality Tracking

**Description:** Track BLAS quality degradation from refits.

**Prerequisites:** T-WGPU-P5.3.1

**Deliverable:** Quality tracking system

**Acceptance Criteria:**
- [ ] quality: f32 (1.0 = fresh, degrades)
- [ ] Degradation rate estimation
- [ ] Threshold for rebuild
- [ ] Metrics exposure

**Estimate:** 4 hours

---

### T-WGPU-P5.3.3 - Dynamic Update Policy

**Description:** Implement policy for update vs rebuild.

**Prerequisites:** T-WGPU-P5.3.2

**Deliverable:** DynamicBlasPolicy struct

**Acceptance Criteria:**
- [ ] max_refit_frames (default: 30)
- [ ] quality_threshold (default: 0.7)
- [ ] Per-mesh policy override
- [ ] Automatic rebuild trigger

**Estimate:** 4 hours

---

### T-WGPU-P5.4.1 - TLAS Instance Descriptor

**Description:** Implement TLAS instance structure.

**Prerequisites:** T-WGPU-P5.1.5

**Deliverable:** TlasInstance struct

**Acceptance Criteria:**
- [ ] BLAS reference
- [ ] Transform matrix (3x4)
- [ ] Instance ID (custom data)
- [ ] Mask (8-bit)
- [ ] SBT offset
- [ ] Instance flags

**Estimate:** 4 hours

---

### T-WGPU-P5.4.2 - TLAS Construction

**Description:** Implement TLAS build from instances.

**Prerequisites:** T-WGPU-P5.4.1

**Deliverable:** `build_tlas()` method

**Acceptance Criteria:**
- [ ] Instance buffer upload
- [ ] Build sizes query
- [ ] Scratch allocation
- [ ] Build command encoding

**Estimate:** 6 hours

---

### T-WGPU-P5.4.3 - TLAS Per-Frame Rebuild

**Description:** Implement efficient per-frame TLAS rebuild.

**Prerequisites:** T-WGPU-P5.4.2

**Deliverable:** Per-frame TLAS update

**Acceptance Criteria:**
- [ ] Transform-only update path
- [ ] Instance buffer double-buffering
- [ ] Minimal rebuild cost
- [ ] Frame timing metrics

**Estimate:** 4 hours

---

### T-WGPU-P5.4.4 - Instance Culling Integration

**Description:** Integrate visibility culling with TLAS.

**Prerequisites:** T-WGPU-P5.4.3

**Deliverable:** Culled TLAS build

**Acceptance Criteria:**
- [ ] Frustum cull instances
- [ ] Occlusion cull (optional)
- [ ] Skip invisible instances
- [ ] Instance count reduction metric

**Estimate:** 4 hours

---

### T-WGPU-P5.5.1 - AS Memory Manager

**Description:** Implement AS memory tracking.

**Prerequisites:** T-WGPU-P5.1.5, T-WGPU-P5.4.2

**Deliverable:** ASMemoryManager struct

**Acceptance Criteria:**
- [ ] Total BLAS memory
- [ ] Total TLAS memory
- [ ] Per-mesh memory
- [ ] Memory report

**Estimate:** 6 hours

---

### T-WGPU-P5.5.2 - AS Memory Budget

**Description:** Implement memory budget for AS.

**Prerequisites:** T-WGPU-P5.5.1

**Deliverable:** Memory budget system

**Acceptance Criteria:**
- [ ] Budget configuration
- [ ] Warning on budget exceeded
- [ ] LOD integration for budget
- [ ] Automatic BLAS eviction (LRU)

**Estimate:** 4 hours

---

### T-WGPU-P5.6.1 - Ray Query Fundamentals

**Description:** Implement ray query setup.

**Prerequisites:** Phase 4 complete

**Deliverable:** Ray query infrastructure

**Acceptance Criteria:**
- [ ] Feature check (ray_query)
- [ ] TLAS binding in compute/fragment
- [ ] Ray query variable declaration
- [ ] Basic trace pattern

**Estimate:** 4 hours

---

### T-WGPU-P5.6.2 - Ray Query WGSL API

**Description:** Implement complete ray query WGSL usage.

**Prerequisites:** T-WGPU-P5.6.1

**Deliverable:** Ray query WGSL patterns

**Acceptance Criteria:**
- [ ] rayQueryInitialize()
- [ ] rayQueryProceed()
- [ ] rayQueryGetCommittedIntersection*()
- [ ] rayQueryGetCandidateIntersection*()
- [ ] rayQueryConfirmIntersection()
- [ ] rayQueryTerminate()

**Estimate:** 6 hours

---

### T-WGPU-P5.6.3 - Ray Flags

**Description:** Implement all ray flags.

**Prerequisites:** T-WGPU-P5.6.2

**Deliverable:** Ray flags documentation and usage

**Acceptance Criteria:**
- [ ] RAY_FLAG_FORCE_OPAQUE
- [ ] RAY_FLAG_TERMINATE_ON_FIRST_HIT
- [ ] RAY_FLAG_CULL_FRONT_FACING
- [ ] RAY_FLAG_CULL_BACK_FACING
- [ ] RAY_FLAG_SKIP_TRIANGLES
- [ ] RAY_FLAG_SKIP_AABBS

**Estimate:** 3 hours

---

### T-WGPU-P5.6.4 - Shadow Ray Pattern

**Description:** Implement shadow ray via ray query.

**Prerequisites:** T-WGPU-P5.6.3

**Deliverable:** Shadow ray shader and pipeline

**Acceptance Criteria:**
- [ ] Early termination (TERMINATE_ON_FIRST_HIT)
- [ ] Binary visibility result
- [ ] Light loop integration
- [ ] Soft shadows (multi-ray)

**Estimate:** 6 hours

---

### T-WGPU-P5.6.5 - AO Ray Pattern

**Description:** Implement ambient occlusion via ray query.

**Prerequisites:** T-WGPU-P5.6.4

**Deliverable:** AO shader and pipeline

**Acceptance Criteria:**
- [ ] Hemisphere sampling
- [ ] Cosine-weighted distribution
- [ ] Max distance parameter
- [ ] Accumulation over frames

**Estimate:** 6 hours

---

### T-WGPU-P5.6.6 - Reflection Ray Pattern

**Description:** Implement reflection via ray query.

**Prerequisites:** T-WGPU-P5.6.4

**Deliverable:** Reflection shader

**Acceptance Criteria:**
- [ ] Reflect ray calculation
- [ ] Closest hit query
- [ ] Hit material sampling
- [ ] Roughness integration (cone tracing)

**Estimate:** 6 hours

---

### T-WGPU-P5.7.1 - RT Pipeline Descriptor

**Description:** Implement RT pipeline creation (experimental).

**Prerequisites:** T-WGPU-P5.6.1

**Deliverable:** RT pipeline creation (feature-gated)

**Acceptance Criteria:**
- [ ] Feature check (ray_tracing_pipeline)
- [ ] Pipeline descriptor struct
- [ ] Max recursion depth
- [ ] Max payload/attribute sizes
- [ ] Pipeline layout

**Estimate:** 6 hours

---

### T-WGPU-P5.7.2 - Ray Gen Shader Stage

**Description:** Implement ray generation shader support.

**Prerequisites:** T-WGPU-P5.7.1

**Deliverable:** Ray gen shader stage

**Acceptance Criteria:**
- [ ] @raygeneration attribute
- [ ] Camera ray generation
- [ ] traceRay() intrinsic
- [ ] Output to image

**Estimate:** 4 hours

---

### T-WGPU-P5.7.3 - Intersection Shader Stage

**Description:** Implement intersection shader for procedural geometry.

**Prerequisites:** T-WGPU-P5.7.2

**Deliverable:** Intersection shader stage

**Acceptance Criteria:**
- [ ] @intersection attribute
- [ ] Hit distance report
- [ ] Custom attributes
- [ ] Procedural examples (sphere, SDF)

**Estimate:** 4 hours

---

### T-WGPU-P5.7.4 - Any-Hit Shader Stage

**Description:** Implement any-hit shader for transparency.

**Prerequisites:** T-WGPU-P5.7.2

**Deliverable:** Any-hit shader stage

**Acceptance Criteria:**
- [ ] @anyhit attribute
- [ ] Alpha testing
- [ ] ignoreIntersection()
- [ ] terminateRay()

**Estimate:** 4 hours

---

### T-WGPU-P5.7.5 - Closest-Hit Shader Stage

**Description:** Implement closest-hit shader for shading.

**Prerequisites:** T-WGPU-P5.7.2

**Deliverable:** Closest-hit shader stage

**Acceptance Criteria:**
- [ ] @closesthit attribute
- [ ] Material lookup
- [ ] Recursive tracing
- [ ] Payload write

**Estimate:** 4 hours

---

### T-WGPU-P5.7.6 - Miss Shader Stage

**Description:** Implement miss shader for environment.

**Prerequisites:** T-WGPU-P5.7.2

**Deliverable:** Miss shader stage

**Acceptance Criteria:**
- [ ] @miss attribute
- [ ] Environment map sampling
- [ ] Sky color fallback
- [ ] Multiple miss shaders (index)

**Estimate:** 3 hours

---

### T-WGPU-P5.7.7 - Callable Shader Stage

**Description:** Implement callable shader for utilities.

**Prerequisites:** T-WGPU-P5.7.2

**Deliverable:** Callable shader stage

**Acceptance Criteria:**
- [ ] @callable attribute
- [ ] executeCallable() intrinsic
- [ ] Use case (BRDF evaluation)
- [ ] Callable data

**Estimate:** 3 hours

---

### T-WGPU-P5.8.1 - Hit Group Definition

**Description:** Define hit group structure.

**Prerequisites:** T-WGPU-P5.7.5

**Deliverable:** HitGroup struct

**Acceptance Criteria:**
- [ ] Triangle hit group (closest + any)
- [ ] Procedural hit group (intersection + closest + any)
- [ ] Group configuration

**Estimate:** 4 hours

---

### T-WGPU-P5.8.2 - SBT Layout

**Description:** Define SBT memory layout.

**Prerequisites:** T-WGPU-P5.8.1

**Deliverable:** SBT layout specification

**Acceptance Criteria:**
- [ ] Ray gen records region
- [ ] Miss records region
- [ ] Hit group records region
- [ ] Callable records region
- [ ] Alignment requirements

**Estimate:** 6 hours

---

### T-WGPU-P5.8.3 - SBT Builder

**Description:** Implement SBT construction.

**Prerequisites:** T-WGPU-P5.8.2

**Deliverable:** SBTBuilder struct

**Acceptance Criteria:**
- [ ] add_ray_gen_record()
- [ ] add_miss_record()
- [ ] add_hit_group_record()
- [ ] add_callable_record()
- [ ] build() -> SBT buffer

**Estimate:** 8 hours

---

### T-WGPU-P5.8.4 - Hit Group Indexing

**Description:** Implement hit group indexing formula.

**Prerequisites:** T-WGPU-P5.8.3

**Deliverable:** Hit group index calculation

**Acceptance Criteria:**
- [ ] Formula: instance_index * sbt_offset + geometry_index
- [ ] Multi-material support
- [ ] Index validation

**Estimate:** 4 hours

---

### T-WGPU-P5.9.1 - RT Dispatch

**Description:** Implement RT pipeline dispatch.

**Prerequisites:** T-WGPU-P5.8.3

**Deliverable:** `dispatch_rays()` wrapper

**Acceptance Criteria:**
- [ ] dispatch_rays(width, height, depth)
- [ ] SBT binding
- [ ] Dimensions validation

**Estimate:** 4 hours

---

### T-WGPU-P5.9.2 - Primary Ray Effect

**Description:** Implement primary ray casting effect.

**Prerequisites:** T-WGPU-P5.9.1

**Deliverable:** Primary ray effect

**Acceptance Criteria:**
- [ ] Camera rays generation
- [ ] Closest hit shading
- [ ] Miss environment
- [ ] Output to render target

**Estimate:** 6 hours

---

### T-WGPU-P5.9.3 - RT Shadows Effect

**Description:** Implement RT shadow effect.

**Prerequisites:** T-WGPU-P5.9.2

**Deliverable:** RT shadows

**Acceptance Criteria:**
- [ ] Shadow ray for each light
- [ ] Soft shadows (area light)
- [ ] Denoising integration hook
- [ ] Performance comparison

**Estimate:** 6 hours

---

### T-WGPU-P5.9.4 - RT Reflections Effect

**Description:** Implement RT reflections effect.

**Prerequisites:** T-WGPU-P5.9.2

**Deliverable:** RT reflections

**Acceptance Criteria:**
- [ ] GBuffer roughness lookup
- [ ] Glossy reflections
- [ ] Multi-bounce (configurable)
- [ ] Fallback to screen-space

**Estimate:** 6 hours

---

### T-WGPU-P5.9.5 - Single Bounce GI

**Description:** Implement single bounce global illumination.

**Prerequisites:** T-WGPU-P5.9.4

**Deliverable:** Single bounce GI

**Acceptance Criteria:**
- [ ] Diffuse ray casting
- [ ] Hemisphere sampling
- [ ] Indirect lighting accumulation
- [ ] Denoising integration hook

**Estimate:** 8 hours

---

### T-WGPU-P5.10.1 - Unit Tests

**Description:** Write unit tests for Phase 5.

**Prerequisites:** All T-WGPU-P5.1-9 tasks

**Deliverable:** Unit tests

**Acceptance Criteria:**
- [ ] BLAS construction tests
- [ ] TLAS construction tests
- [ ] SBT layout tests
- [ ] Ray query tests
- [ ] 80%+ coverage

**Estimate:** 8 hours

---

### T-WGPU-P5.10.2 - Integration Tests

**Description:** Write integration tests for RT.

**Prerequisites:** T-WGPU-P5.10.1

**Deliverable:** Integration tests

**Acceptance Criteria:**
- [ ] AS build pipeline test
- [ ] Shadow tracing test
- [ ] Compaction test
- [ ] Multi-frame TLAS test

**Estimate:** 8 hours

---

### T-WGPU-P5.10.3 - Visual Tests

**Description:** Create visual verification tests.

**Prerequisites:** T-WGPU-P5.10.2

**Deliverable:** Visual tests

**Acceptance Criteria:**
- [ ] RT shadow reference image
- [ ] RT reflection reference image
- [ ] AO reference image
- [ ] Image comparison threshold

**Estimate:** 6 hours

---

## Task Dependencies

```
Phase 4 Complete
    |
    +---> T-WGPU-P5.1.1 (Triangle geometry)
              |
              +---> T-WGPU-P5.1.2, P5.1.3
              +---> T-WGPU-P5.1.4 (Scratch pool)
                        |
                        +---> T-WGPU-P5.1.5 (BLAS construction)
                                  |
                                  +---> T-WGPU-P5.2.* (Compaction)
                                  +---> T-WGPU-P5.3.* (Update)
                                  +---> T-WGPU-P5.4.* (TLAS)
                                  +---> T-WGPU-P5.5.* (Memory)

    +---> T-WGPU-P5.6.1 (Ray query fundamentals)
              |
              +---> T-WGPU-P5.6.2 through P5.6.6

    +---> T-WGPU-P5.7.1 (RT pipeline, experimental)
              |
              +---> T-WGPU-P5.7.2 through P5.7.7 (Stages)
              +---> T-WGPU-P5.8.* (SBT)
              +---> T-WGPU-P5.9.* (Effects)

All --> T-WGPU-P5.10.1 --> T-WGPU-P5.10.2 --> T-WGPU-P5.10.3
```

---

*End of PHASE_5_RAY_TRACING_TODO.md*
