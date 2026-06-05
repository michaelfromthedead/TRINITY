# PHASE 2: RESOURCES - Task List

**Phase:** 2 - RESOURCES
**Estimated Duration:** 3-4 weeks
**Task ID Prefix:** T-WGPU-P2

---

## Task Summary

| ID | Task | Est. Hours | Status |
|----|------|------------|--------|
| T-WGPU-P2.1.1 | Buffer creation API | 4 | - |
| T-WGPU-P2.1.2 | Buffer usage flags | 3 | - |
| T-WGPU-P2.1.3 | Buffer mapping | 6 | - |
| T-WGPU-P2.1.4 | Vertex buffer registry | 4 | - |
| T-WGPU-P2.1.5 | Dynamic uniform buffers | 4 | - |
| T-WGPU-P2.1.6 | Storage buffers | 3 | - |
| T-WGPU-P2.1.7 | Indirect buffers | 4 | - |
| T-WGPU-P2.2.1 | Buffer pool | 6 | - |
| T-WGPU-P2.2.2 | Ring buffer | 6 | - |
| T-WGPU-P2.2.3 | Deferred destroyer | 4 | - |
| T-WGPU-P2.3.1 | Texture creation | 4 | - |
| T-WGPU-P2.3.2 | Texture formats | 6 | - |
| T-WGPU-P2.3.3 | Texture views | 4 | - |
| T-WGPU-P2.3.4 | Mip generation | 6 | - |
| T-WGPU-P2.3.5 | Texture uploads | 4 | - |
| T-WGPU-P2.4.1 | Sampler creation | 3 | - |
| T-WGPU-P2.4.2 | Sampler cache | 4 | - |
| T-WGPU-P2.5.1 | Bind group layout cache | 4 | - |
| T-WGPU-P2.5.2 | Bind group creation | 4 | - |
| T-WGPU-P2.5.3 | Pipeline layout | 4 | - |
| T-WGPU-P2.5.4 | Push constants | 3 | - |
| T-WGPU-P2.6.1 | Bindless texture registry | 6 | - |
| T-WGPU-P2.6.2 | Bindless buffer registry | 4 | - |
| T-WGPU-P2.6.3 | Index allocator | 4 | - |
| T-WGPU-P2.7.1 | Shader module creation | 4 | - |
| T-WGPU-P2.7.2 | Shader cache | 4 | - |
| T-WGPU-P2.7.3 | Naga validation | 4 | - |
| T-WGPU-P2.7.4 | Shader reflection | 6 | - |
| T-WGPU-P2.7.5 | Override constants | 4 | - |
| T-WGPU-P2.7.6 | Permutation manager | 6 | - |
| T-WGPU-P2.7.7 | Shader hot-reload | 8 | - |
| T-WGPU-P2.8.1 | Unit tests | 10 | - |
| T-WGPU-P2.8.2 | Integration tests | 8 | - |

**Total Estimated Hours:** 154 hours

---

## Detailed Tasks

### T-WGPU-P2.1.1 - Buffer Creation API

**Description:** Implement core buffer creation with descriptor.

**Prerequisites:** Phase 1 complete

**Deliverable:** `create_buffer()` in resources/buffer.rs

**Acceptance Criteria:**
- [ ] BufferDescriptor with label, size, usage, mapped_at_creation
- [ ] Returns TrinityBuffer wrapper
- [ ] Size alignment validation (4 bytes)
- [ ] Logs allocation

**Estimate:** 4 hours

---

### T-WGPU-P2.1.2 - Buffer Usage Flags

**Description:** Implement BufferUsages configuration and validation.

**Prerequisites:** T-WGPU-P2.1.1

**Deliverable:** Usage flag helpers and validation

**Acceptance Criteria:**
- [ ] All 10 usage flags exposed
- [ ] Common combinations documented (vertex, uniform, staging)
- [ ] Invalid combinations rejected (MAP_READ + VERTEX)
- [ ] Usage table in docs

**Estimate:** 3 hours

---

### T-WGPU-P2.1.3 - Buffer Mapping

**Description:** Implement sync and async buffer mapping.

**Prerequisites:** T-WGPU-P2.1.1

**Deliverable:** Mapping API with async wrapper

**Acceptance Criteria:**
- [ ] Sync mapping via mapped_at_creation
- [ ] Async mapping via map_async
- [ ] TRINITY async wrapper with oneshot channel
- [ ] Unmap on drop or explicit call
- [ ] Read/write mode selection

**Estimate:** 6 hours

---

### T-WGPU-P2.1.4 - Vertex Buffer Registry

**Description:** Implement VertexFormatRegistry with standard layouts.

**Prerequisites:** T-WGPU-P2.1.1

**Deliverable:** VertexFormatRegistry struct

**Acceptance Criteria:**
- [ ] Standard PBR layout (48 bytes: pos, normal, uv, tangent)
- [ ] Skinned mesh layout (72 bytes: + joints, weights)
- [ ] Terrain layout (32 bytes)
- [ ] Particle layout (32 bytes)
- [ ] UI layout (20 bytes)
- [ ] vertex_attr_array! macro usage
- [ ] Layout by ID lookup

**Estimate:** 4 hours

---

### T-WGPU-P2.1.5 - Dynamic Uniform Buffers

**Description:** Implement dynamic uniform buffer support with offsets.

**Prerequisites:** T-WGPU-P2.1.1

**Deliverable:** Dynamic uniform buffer helpers

**Acceptance Criteria:**
- [ ] UNIFORM_ALIGNMENT constant (256 bytes)
- [ ] Offset calculation helper
- [ ] Per-object transform binding
- [ ] Dynamic offset in set_bind_group

**Estimate:** 4 hours

---

### T-WGPU-P2.1.6 - Storage Buffers

**Description:** Implement storage buffer configuration.

**Prerequisites:** T-WGPU-P2.1.1

**Deliverable:** Storage buffer binding helpers

**Acceptance Criteria:**
- [ ] Read-only storage buffer
- [ ] Read-write storage buffer
- [ ] Minimum binding size constraint
- [ ] BindingType::Buffer configuration

**Estimate:** 3 hours

---

### T-WGPU-P2.1.7 - Indirect Buffers

**Description:** Implement indirect draw/dispatch buffer structs.

**Prerequisites:** T-WGPU-P2.1.1

**Deliverable:** DrawIndirectArgs, DispatchIndirectArgs structs

**Acceptance Criteria:**
- [ ] DrawIndirectArgs (16 bytes)
- [ ] DrawIndexedIndirectArgs (20 bytes, note: base_vertex is i32)
- [ ] DispatchIndirectArgs (12 bytes)
- [ ] bytemuck derives
- [ ] Usage: INDIRECT

**Estimate:** 4 hours

---

### T-WGPU-P2.2.1 - Buffer Pool

**Description:** Implement buffer pool with size classes.

**Prerequisites:** T-WGPU-P2.1.1

**Deliverable:** BufferPool struct

**Acceptance Criteria:**
- [ ] Size classes: 256B, 1KB, 4KB, 16KB, 64KB, 256KB, 1MB
- [ ] Free list per class
- [ ] Acquire/release API
- [ ] Growth policy (double on exhaust)
- [ ] Shrink policy (release if >50% free)

**Estimate:** 6 hours

---

### T-WGPU-P2.2.2 - Ring Buffer

**Description:** Implement ring buffer for per-frame data.

**Prerequisites:** T-WGPU-P2.1.1

**Deliverable:** RingBuffer struct

**Acceptance Criteria:**
- [ ] Triple-buffered (3 frames)
- [ ] Per-frame offset tracking
- [ ] Wrap-around with fence waiting
- [ ] allocate() returns offset
- [ ] Frame reset on begin_frame()
- [ ] Metrics: utilization, wrap count

**Estimate:** 6 hours

---

### T-WGPU-P2.2.3 - Deferred Destroyer

**Description:** Implement deferred resource destruction.

**Prerequisites:** T-WGPU-P2.1.1

**Deliverable:** DeferredDestroyer struct

**Acceptance Criteria:**
- [ ] Queue destruction for frame + N (default 2)
- [ ] Batch destroy on frame end
- [ ] Supports buffers and textures
- [ ] Cleanup on drop

**Estimate:** 4 hours

---

### T-WGPU-P2.3.1 - Texture Creation

**Description:** Implement texture creation with descriptor.

**Prerequisites:** Phase 1 complete

**Deliverable:** `create_texture()` in resources/texture.rs

**Acceptance Criteria:**
- [ ] TextureDescriptor with all fields
- [ ] view_formats for reinterpretation
- [ ] Returns TrinityTexture with view
- [ ] Logs allocation with size estimate

**Estimate:** 4 hours

---

### T-WGPU-P2.3.2 - Texture Formats

**Description:** Implement TextureFormatSelector with platform awareness.

**Prerequisites:** T-WGPU-P2.3.1

**Deliverable:** TextureFormatSelector struct

**Acceptance Criteria:**
- [ ] `color_attachment()` - sRGB on Windows, linear on Metal
- [ ] `depth()` - Depth32Float primary, Depth24Plus fallback
- [ ] `normal_map()` - RG16 or RGBA8 based on support
- [ ] `compressed()` - BC on Windows/Linux, ASTC on Apple/Mobile
- [ ] Format tables for all categories
- [ ] Platform detection

**Estimate:** 6 hours

---

### T-WGPU-P2.3.3 - Texture Views

**Description:** Implement texture view creation with dimension/format options.

**Prerequisites:** T-WGPU-P2.3.1

**Deliverable:** View creation helpers

**Acceptance Criteria:**
- [ ] Mip-level subrange view
- [ ] Array layer subrange view
- [ ] Aspect selection (color, depth, stencil)
- [ ] Format reinterpretation (sRGB toggle)
- [ ] Dimension conversion (2D -> CubeArray)

**Estimate:** 4 hours

---

### T-WGPU-P2.3.4 - Mip Generation

**Description:** Implement compute shader mip generation.

**Prerequisites:** T-WGPU-P2.3.1, T-WGPU-P2.7.1

**Deliverable:** MipGenerator struct

**Acceptance Criteria:**
- [ ] calculate_mip_count() function
- [ ] Compute shader mip generation
- [ ] Blit-based fallback
- [ ] sRGB correct (linearize, downsample, gamma)
- [ ] Handles 1D, 2D, 3D, arrays

**Estimate:** 6 hours

---

### T-WGPU-P2.3.5 - Texture Uploads

**Description:** Implement CPU -> GPU texture uploads.

**Prerequisites:** T-WGPU-P2.3.1

**Deliverable:** Texture upload API

**Acceptance Criteria:**
- [ ] write_texture() for direct upload
- [ ] Staging buffer for large uploads
- [ ] Row pitch alignment (256 bytes)
- [ ] Subregion uploads
- [ ] Format conversion helpers

**Estimate:** 4 hours

---

### T-WGPU-P2.4.1 - Sampler Creation

**Description:** Implement sampler creation with full descriptor.

**Prerequisites:** Phase 1 complete

**Deliverable:** `create_sampler()` in resources/sampler.rs

**Acceptance Criteria:**
- [ ] SamplerDescriptor with all fields
- [ ] Address modes: ClampToEdge, Repeat, MirrorRepeat
- [ ] Filter modes: Nearest, Linear
- [ ] Anisotropy support (where available)
- [ ] Comparison sampler for shadows

**Estimate:** 3 hours

---

### T-WGPU-P2.4.2 - Sampler Cache

**Description:** Implement sampler caching with presets.

**Prerequisites:** T-WGPU-P2.4.1

**Deliverable:** SamplerCache struct

**Acceptance Criteria:**
- [ ] Hash-based lookup by descriptor
- [ ] Arc<Sampler> for shared ownership
- [ ] Presets: linear_repeat, linear_clamp, point, shadow
- [ ] get_or_create() API
- [ ] Metrics: cache size, hit rate

**Estimate:** 4 hours

---

### T-WGPU-P2.5.1 - Bind Group Layout Cache

**Description:** Implement bind group layout caching.

**Prerequisites:** Phase 1 complete

**Deliverable:** LayoutCache struct

**Acceptance Criteria:**
- [ ] Key: sorted binding entries hash
- [ ] Value: Arc<BindGroupLayout>
- [ ] get_or_create() API
- [ ] Layout compatibility checking

**Estimate:** 4 hours

---

### T-WGPU-P2.5.2 - Bind Group Creation

**Description:** Implement bind group creation with caching.

**Prerequisites:** T-WGPU-P2.5.1

**Deliverable:** BindGroupCache struct

**Acceptance Criteria:**
- [ ] BindGroupDescriptor construction
- [ ] Cache by layout + resource handles
- [ ] Invalidation on resource destruction
- [ ] Supports all binding types

**Estimate:** 4 hours

---

### T-WGPU-P2.5.3 - Pipeline Layout

**Description:** Implement pipeline layout creation.

**Prerequisites:** T-WGPU-P2.5.1

**Deliverable:** Pipeline layout helpers

**Acceptance Criteria:**
- [ ] PipelineLayoutDescriptor with bind group layouts
- [ ] Push constant ranges (where supported)
- [ ] Layout caching
- [ ] TRINITY standard layouts (global, material, object, bindless)

**Estimate:** 4 hours

---

### T-WGPU-P2.5.4 - Push Constants

**Description:** Implement push constant support.

**Prerequisites:** T-WGPU-P2.5.3

**Deliverable:** Push constant helpers

**Acceptance Criteria:**
- [ ] Feature check (PUSH_CONSTANTS)
- [ ] Range configuration (stages, offset, size)
- [ ] set_push_constants() wrapper
- [ ] Fallback to uniform buffer on unsupported

**Estimate:** 3 hours

---

### T-WGPU-P2.6.1 - Bindless Texture Registry

**Description:** Implement bindless texture array management.

**Prerequisites:** T-WGPU-P2.5.1, T-WGPU-P2.3.1

**Deliverable:** TextureRegistry struct

**Acceptance Criteria:**
- [ ] Feature check (TEXTURE_BINDING_ARRAY)
- [ ] Slot allocation for textures
- [ ] Free slot recycling
- [ ] Bind group with count
- [ ] Max textures capped to limit

**Estimate:** 6 hours

---

### T-WGPU-P2.6.2 - Bindless Buffer Registry

**Description:** Implement bindless buffer array management.

**Prerequisites:** T-WGPU-P2.5.1, T-WGPU-P2.1.1

**Deliverable:** BufferRegistry struct

**Acceptance Criteria:**
- [ ] Storage buffer array binding
- [ ] Slot allocation for buffers
- [ ] Dirty range tracking
- [ ] Free slot recycling

**Estimate:** 4 hours

---

### T-WGPU-P2.6.3 - Index Allocator

**Description:** Implement generic index allocator with free list.

**Prerequisites:** None (utility)

**Deliverable:** IndexAllocator struct

**Acceptance Criteria:**
- [ ] allocate() returns next free index
- [ ] free(index) returns to free list
- [ ] free_indices Vec for recycling
- [ ] Generation tracking (optional, for validation)

**Estimate:** 4 hours

---

### T-WGPU-P2.7.1 - Shader Module Creation

**Description:** Implement shader module creation from WGSL.

**Prerequisites:** Phase 1 complete

**Deliverable:** `create_shader_module()` in shaders/mod.rs

**Acceptance Criteria:**
- [ ] WGSL source input (ShaderSource::Wgsl)
- [ ] SPIR-V input (create_shader_module_spirv, unsafe)
- [ ] Label for debugging
- [ ] Error handling with location info

**Estimate:** 4 hours

---

### T-WGPU-P2.7.2 - Shader Cache

**Description:** Implement shader module caching.

**Prerequisites:** T-WGPU-P2.7.1

**Deliverable:** ShaderCache struct

**Acceptance Criteria:**
- [ ] In-memory HashMap cache
- [ ] Key: shader path or content hash
- [ ] get_or_compile() API
- [ ] Invalidation API for hot-reload
- [ ] Optional disk cache for SPIR-V

**Estimate:** 4 hours

---

### T-WGPU-P2.7.3 - Naga Validation

**Description:** Implement Naga pre-validation for better errors.

**Prerequisites:** T-WGPU-P2.7.1

**Deliverable:** Naga validation layer

**Acceptance Criteria:**
- [ ] naga::front::wgsl::parse_str()
- [ ] naga::valid::Validator with flags
- [ ] Error location extraction
- [ ] Human-readable error messages
- [ ] Source code snippet in error

**Estimate:** 4 hours

---

### T-WGPU-P2.7.4 - Shader Reflection

**Description:** Implement shader reflection from Naga IR.

**Prerequisites:** T-WGPU-P2.7.3

**Deliverable:** ShaderReflection struct

**Acceptance Criteria:**
- [ ] Entry point enumeration
- [ ] Binding extraction (group, binding, type)
- [ ] Resource type detection
- [ ] Automatic layout generation (optional)
- [ ] Push constant reflection

**Estimate:** 6 hours

---

### T-WGPU-P2.7.5 - Override Constants

**Description:** Implement pipeline override constants.

**Prerequisites:** T-WGPU-P2.7.1

**Deliverable:** Override constant support

**Acceptance Criteria:**
- [ ] WGSL @id(N) override constants
- [ ] PipelineCompilationOptions.constants
- [ ] Bool/numeric constant types
- [ ] Constant lookup by name or ID

**Estimate:** 4 hours

---

### T-WGPU-P2.7.6 - Permutation Manager

**Description:** Implement shader permutation management.

**Prerequisites:** T-WGPU-P2.7.2, T-WGPU-P2.7.5

**Deliverable:** ShaderPermutationManager struct

**Acceptance Criteria:**
- [ ] FeatureFlags bitflags (7 common: SKINNED, ALPHA_TEST, etc.)
- [ ] Permutation key generation
- [ ] Lazy compilation
- [ ] Permutation cache
- [ ] Max permutation limit

**Estimate:** 6 hours

---

### T-WGPU-P2.7.7 - Shader Hot-Reload

**Description:** Implement shader hot-reload for development.

**Prerequisites:** T-WGPU-P2.7.2

**Deliverable:** ShaderHotReload struct

**Acceptance Criteria:**
- [ ] notify crate for file watching
- [ ] Watch shader directories
- [ ] Channel-based reload notification
- [ ] Module rebuild on change
- [ ] Pipeline invalidation callback
- [ ] Error reporting without crash
- [ ] Debug-only compilation

**Estimate:** 8 hours

---

### T-WGPU-P2.8.1 - Unit Tests

**Description:** Write unit tests for all Phase 2 components.

**Prerequisites:** All T-WGPU-P2.1-7 tasks

**Deliverable:** Tests in resources/tests/, shaders/tests/

**Acceptance Criteria:**
- [ ] Buffer allocation tests
- [ ] Ring buffer wrap tests
- [ ] Texture format selection tests
- [ ] Sampler cache deduplication tests
- [ ] Bind group layout hash tests
- [ ] Bindless allocation tests
- [ ] Shader cache tests
- [ ] Permutation key tests
- [ ] 80%+ coverage

**Estimate:** 10 hours

---

### T-WGPU-P2.8.2 - Integration Tests

**Description:** Write integration tests for resource operations.

**Prerequisites:** T-WGPU-P2.8.1

**Deliverable:** Integration tests

**Acceptance Criteria:**
- [ ] Buffer upload test
- [ ] Texture upload with mips test
- [ ] Bind group creation test
- [ ] Shader compilation test
- [ ] Hot-reload test (simulated file change)
- [ ] Bindless allocation/free cycle test

**Estimate:** 8 hours

---

## Task Dependencies

```
Phase 1 Complete
    |
    +---> T-WGPU-P2.1.1 (Buffer creation)
    |         |
    |         +---> T-WGPU-P2.1.2, P2.1.3, P2.1.4, P2.1.5, P2.1.6, P2.1.7
    |         +---> T-WGPU-P2.2.1 (Pool)
    |         +---> T-WGPU-P2.2.2 (Ring)
    |         +---> T-WGPU-P2.2.3 (Deferred)
    |
    +---> T-WGPU-P2.3.1 (Texture creation)
    |         |
    |         +---> T-WGPU-P2.3.2, P2.3.3, P2.3.5
    |         +---> T-WGPU-P2.3.4 (Mips, needs P2.7.1)
    |
    +---> T-WGPU-P2.4.1 --> T-WGPU-P2.4.2 (Samplers)
    |
    +---> T-WGPU-P2.5.1 (Layout cache)
              |
              +---> T-WGPU-P2.5.2, P2.5.3
              +---> T-WGPU-P2.5.4 (Push constants)
              +---> T-WGPU-P2.6.1, P2.6.2, P2.6.3 (Bindless)

T-WGPU-P2.7.1 (Shader module)
    |
    +---> T-WGPU-P2.7.2 (Cache)
    +---> T-WGPU-P2.7.3 (Naga validation)
              |
              +---> T-WGPU-P2.7.4 (Reflection)
    +---> T-WGPU-P2.7.5 (Override constants)
    +---> T-WGPU-P2.7.6 (Permutations, needs P2.7.2, P2.7.5)
    +---> T-WGPU-P2.7.7 (Hot-reload, needs P2.7.2)

All --> T-WGPU-P2.8.1 --> T-WGPU-P2.8.2
```

---

*End of PHASE_2_RESOURCES_TODO.md*
