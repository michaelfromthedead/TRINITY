# PHASE 2: RESOURCES - Architecture

**Scope:** Buffers, textures, samplers, bind groups, shaders
**Duration:** 3-4 weeks
**Dependencies:** Phase 1 (CORE)
**Produces:** Complete resource management layer

---

## Overview

Phase 2 implements the resource management layer including all GPU memory objects, bindings, and shader compilation. This layer provides the building blocks for pipelines.

### Covered Content (from MASTER.md Parts II-III)

- Chapter 2: Buffers
  - 2.1 Buffer fundamentals (usage flags, mapping, destruction)
  - 2.2 Buffer types (vertex, index, uniform, storage, indirect, staging)
  - 2.3 Memory management (suballocation, pooling, ring buffers)

- Chapter 3: Textures
  - 3.1 Fundamentals (dimensions, formats, usage, mip levels)
  - 3.2 Texture formats (uncompressed, float, sRGB, depth, compressed)
  - 3.3 Texture views (dimension conversion, format reinterpretation)
  - 3.4 Samplers (address modes, filters, comparison)
  - 3.5 Texture operations (uploads, copies, mip generation)

- Chapter 4: Bind Groups & Layouts
  - 4.1 Binding model (bind group concept, layout as contract)
  - 4.2 Buffer bindings (uniform, storage, dynamic offsets)
  - 4.3 Texture & sampler bindings
  - 4.4 Advanced binding patterns (bindless via storage/texture arrays)
  - 4.5 Pipeline layouts (creation, push constants, compatibility)

- Chapter 5: WGSL & Naga
  - 5.1 WGSL language (types, address spaces, attributes, built-ins)
  - 5.2 Naga compiler (pipeline, targets, caching)
  - 5.3 Shader modules (creation, error handling, reflection)
  - 5.4 Shader specialization (override constants, permutations, variants)

---

## Architectural Decisions

### ADR-004: Buffer Memory Strategy

**Context:** GPU memory allocation is expensive; frequent small allocations harm performance.

**Decision:** Implement hierarchical allocation:
1. Large persistent buffers: Direct allocation
2. Medium transient buffers: Pool with size classes
3. Small dynamic data: Ring buffer suballocation

**Rationale:** Balances memory efficiency with allocation speed.

**Consequences:**
- Ring buffer for per-frame uniform data
- Pool maintains free lists per size class
- Large allocations bypass pool

---

### ADR-005: Texture Format Selection

**Context:** Texture formats vary by platform; compressed formats differ.

**Decision:** Implement TextureFormatSelector with platform-aware fallbacks:
- BC formats preferred on Windows/Linux
- ASTC formats preferred on mobile/Apple
- ETC2 as fallback for older mobile

**Rationale:** Optimal memory usage without runtime format conversion.

**Consequences:**
- Asset pipeline must provide multiple format variants
- Format selection at load time, not runtime
- Uncompressed fallback always available

---

### ADR-006: Sampler Caching

**Context:** Samplers are immutable and frequently reused.

**Decision:** Implement SamplerCache with hash-based deduplication:
- Key: SamplerDescriptor hash
- Value: Arc<wgpu::Sampler>
- Preset samplers for common configurations

**Rationale:** Avoids duplicate sampler objects; reduces descriptor usage.

**Consequences:**
- Sampler creation goes through cache
- Preset samplers (linear_repeat, linear_clamp, point, shadow) always available
- Arc allows shared ownership

---

### ADR-007: Bindless Architecture

**Context:** Material diversity requires dynamic texture/buffer binding without pipeline rebinding.

**Decision:** Implement bindless via:
1. Texture arrays with TEXTURE_BINDING_ARRAY feature
2. Storage buffer indirection for material data
3. IndexAllocator with free list recycling

**Rationale:** Enables massive material counts with minimal rebinds.

**Consequences:**
- Requires Advanced capability tier
- Fallback to traditional binding on lower tiers
- Index management overhead

---

### ADR-008: Shader Hot-Reload

**Context:** Rapid iteration requires shader changes without restart.

**Decision:** Implement ShaderHotReload with:
- notify crate for file watching
- Channel-based reload notification
- Module rebuild and pipeline invalidation

**Rationale:** Developer productivity; essential for shader development.

**Consequences:**
- Only enabled in debug builds
- Pipeline cache must support invalidation
- Shader errors caught and reported without crash

---

## Component Breakdown

### 1. Buffer Management

```
TrinityBufferSystem
├── allocator: BufferAllocator
├── pools: HashMap<SizeClass, BufferPool>
├── ring_buffer: RingBuffer
├── deferred_destroyer: DeferredDestroyer
└── frame_count: u64
```

**BufferAllocator:**
- `create_buffer()` - Direct allocation
- `allocate_subbuffer()` - Ring buffer allocation

**BufferPool:**
- Size classes: 256B, 1KB, 4KB, 16KB, 64KB, 256KB, 1MB
- Free list per class
- Growth policy

**RingBuffer:**
- Triple-buffered (3 frames in flight)
- Frame-based offset tracking
- Wrap-around with fence waiting

**DeferredDestroyer:**
- Queue destruction for frame + 2
- Batch destroy on frame end

### 2. Texture Management

```
TrinityTextureSystem
├── format_selector: TextureFormatSelector
├── mip_generator: MipGenerator
├── sampler_cache: SamplerCache
└── transient_pool: TexturePool
```

**TextureFormatSelector:**
- `color_attachment()` - sRGB vs linear selection
- `depth()` - Depth format by precision need
- `normal_map()` - Unsigned or signed formats
- Platform detection for compressed formats

**MipGenerator:**
- Compute shader mip generation
- Blit-based fallback
- Format-aware (sRGB correct)

**SamplerCache:**
- Hash-based lookup
- Preset samplers: linear_repeat, linear_clamp, point, shadow
- Reference counting via Arc

### 3. Bind Group Management

```
TrinityBindingSystem
├── layout_cache: LayoutCache
├── bind_group_cache: BindGroupCache
├── bindless_manager: Option<BindlessManager>
└── push_constant_support: bool
```

**LayoutCache:**
- Key: sorted binding entries
- Value: Arc<BindGroupLayout>
- Pipeline layout caching

**BindGroupCache:**
- Key: layout + resource handles
- Value: BindGroup
- Invalidation on resource destruction

**BindlessManager:**
- TextureRegistry: slot allocation for textures
- BufferRegistry: slot allocation for buffers
- MaterialTable: combined texture + buffer indices
- IndexAllocator: free list recycling

### 4. Shader System

```
TrinityShaderSystem
├── cache: ShaderCache
├── hot_reload: Option<ShaderHotReload>
├── permutation_manager: ShaderPermutationManager
├── variant_system: ShaderVariantSystem
└── reflection: ShaderReflection
```

**ShaderCache:**
- In-memory HashMap for compiled modules
- Optional disk cache for SPIR-V
- Naga validation before creation

**ShaderHotReload:**
- notify watcher on shader directories
- Pending reload queue
- Module rebuild callback

**ShaderPermutationManager:**
- FeatureFlags bitflags (7 common flags)
- Permutation key generation
- Lazy compilation

**ShaderVariantSystem:**
- Registry of variant configurations
- `precompile_common_variants()` at startup
- Background compilation

---

## Module Structure

```
crates/renderer-backend/src/
├── resources/
│   ├── mod.rs              # Module exports
│   ├── buffer.rs           # BufferAllocator, BufferPool
│   ├── ring_buffer.rs      # RingBuffer
│   ├── texture.rs          # TextureFormatSelector, MipGenerator
│   ├── sampler.rs          # SamplerCache
│   ├── bind_group.rs       # LayoutCache, BindGroupCache
│   ├── bindless.rs         # BindlessManager, registries
│   └── deferred.rs         # DeferredDestroyer
│
├── shaders/
│   ├── mod.rs              # Module exports
│   ├── cache.rs            # ShaderCache
│   ├── hot_reload.rs       # ShaderHotReload
│   ├── permutations.rs     # ShaderPermutationManager
│   ├── variants.rs         # ShaderVariantSystem
│   ├── reflection.rs       # ShaderReflection from Naga
│   └── wgsl/
│       ├── common/         # Shared WGSL includes
│       ├── vertex/         # Vertex shaders
│       ├── fragment/       # Fragment shaders
│       └── compute/        # Compute shaders
```

---

## Testing Strategy

### Unit Tests

1. **Buffer allocation** - Size classes, suballocation, wrap-around
2. **Ring buffer** - Frame cycling, offset calculation
3. **Texture format** - Platform format selection
4. **Sampler cache** - Deduplication, preset retrieval
5. **Bind group** - Layout compatibility, caching
6. **Bindless** - Index allocation/recycling
7. **Shader cache** - Hit/miss, invalidation
8. **Permutations** - Key generation, lookup

### Integration Tests

1. **Buffer upload** - Staging -> GPU via copy
2. **Texture upload** - CPU -> GPU with mip generation
3. **Bind group creation** - Full pipeline layout
4. **Shader compilation** - WGSL -> Module -> Pipeline
5. **Hot-reload** - File change -> module update

### Blackbox Tests

1. **Memory tracking** - Allocation count, total size
2. **Sampler enumeration** - List cached samplers
3. **Shader reflection** - Dump binding info

---

## Performance Considerations

1. **Buffer Suballocation** - Minimize driver allocations
2. **Ring Buffer** - Zero-wait per-frame updates
3. **Sampler Cache** - Eliminate duplicate samplers
4. **Layout Cache** - Reduce pipeline layout creation
5. **Shader Cache** - Avoid recompilation
6. **Bindless** - Reduce bind group switching

---

## Dependencies

### External Crates

- `wgpu` - Core GPU abstraction
- `naga` - Shader compilation (via wgpu)
- `notify` - File watching for hot-reload
- `bytemuck` - Safe byte casting
- `rustc-hash` - Fast hashing for caches

### Internal Dependencies

- Phase 1: TrinityDevice, TrinityQueue, CapabilityManager

---

## Deliverables Checklist

- [ ] BufferAllocator with pool and ring buffer
- [ ] DeferredDestroyer for safe deletion
- [ ] TextureFormatSelector with platform awareness
- [ ] MipGenerator compute shader
- [ ] SamplerCache with presets
- [ ] LayoutCache and BindGroupCache
- [ ] BindlessManager (conditional on tier)
- [ ] ShaderCache with Naga validation
- [ ] ShaderHotReload (debug only)
- [ ] ShaderPermutationManager
- [ ] Vertex format registry
- [ ] Unit tests (80%+ coverage)
- [ ] Integration tests
- [ ] Documentation

---

*End of PHASE_2_RESOURCES_ARCH.md*
