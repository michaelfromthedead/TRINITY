# GAPSET_12_ASSETS -- Task Definitions

> **TASK_ID format**: T-AS-{PHASE}.{N} where AS = Asset Pipeline
> **Total**: 40 tasks across 6 phases
> **Status**: All [ ] not started

---

## Phase 1: glTF Mesh Pipeline (7 tasks)

### T-AS-1.1 -- glTF 2.0 Parser Implementation
- **Description**: Implement the core glTF 2.0 parser: JSON manifest parsing, schema validation, buffer resolution (base64 + external .bin), accessor slicing, node hierarchy extraction.
- **Acceptance Criteria**:
  - Parses glTF Basic, PBR, and Skin variants from the Khronos sample models
  - Validates required accessor properties, buffer view bounds, animation channel validity
  - Resolves both embedded base64 data URIs and external .bin files
  - Supports streaming parsing for files >2GB without loading full JSON
  - Worker-thread offloadable for JSON parsing and binary decoding
  - Implements progressive loading: skeleton/bounds first, then vertices, then skinning data
- **Dependencies**: S15 math.rs (Vec3/4, Mat4, AABB), gltf crate
- **Estimated Effort**: MEDIUM (3-4 days)

### T-AS-1.2 -- Vertex Format Conversion Engine
- **Description**: Implement the vertex format conversion system: glTF semantics to engine semantics, Interleaved/Split/Compressed layouts, attribute quantization, @import_settings configuration.
- **Acceptance Criteria**:
  - Converts glTF semantics (POSITION, NORMAL, TANGENT, TEXCOORD_0, COLOR_0, JOINTS_0, WEIGHTS_0) to engine-internal representation
  - Produces Interleaved (single buffer), Split (per-attribute buffers), and Compressed layouts
  - Compressed positions: 16-bit float or 10-10-10-2 normalized
  - Compressed normals/tangents: 8-bit SNORM or octahedral (2x16-bit)
  - Compressed UVs: 16-bit float or 16-bit normalized
  - Compressed colors: 8-bit UNORM or 10-10-10-2 HDR
  - @import_settings controls scale, axis_conversion, merge_meshes
- **Dependencies**: T-AS-1.1, S14 RHI buffer interface
- **Estimated Effort**: MEDIUM (2-3 days)

### T-AS-1.3 -- Index Buffer Optimization Pipeline
- **Description**: Implement index buffer type selection (16/32/8-bit), cache-optimized reordering, pre-transform cache optimization, and stripification.
- **Acceptance Criteria**:
  - Selects smallest compatible index type: 16-bit (<65536 verts), 32-bit (>=65536), 8-bit (meshlets)
  - Applies meshopt or Tom Forsyth algorithm for post-transform cache optimization
  - Applies pre-transform vertex cache reordering
  - Stripification (triangle strips) optional per-RHI
  - All optimizations idempotent: same input -> same output
- **Dependencies**: T-AS-1.2, meshopt crate
- **Estimated Effort**: SMALL (1-2 days)

### T-AS-1.4 -- Meshlet Generation System
- **Description**: Implement meshlet partitioning, optimization, bounds computation, and adjacency linking for GPU-driven culling.
- **Acceptance Criteria**:
  - Partitions mesh into ~64 vertex, ~124 triangle clusters (Morton order or k-means)
  - Computes bounding sphere for each meshlet (frustum + occlusion culling)
  - Computes cone of normals for backface culling
  - Computes adjacency information for seam-free LOD transitions
  - Optional: coarse depth interval for Hi-Z culling
  - Reorders vertices/indices within each cluster for GPU cache
  - Idempotent: same mesh -> same meshlet set
- **Dependencies**: T-AS-1.3, meshopt crate
- **Estimated Effort**: MEDIUM (3-4 days)

### T-AS-1.5 -- BLAS Baking Pipeline
- **Description**: Implement bottom-level acceleration structure baking: build input preparation, compaction, serialization, and flag configuration.
- **Acceptance Criteria**:
  - Produces BLAS build input from GPU-native vertex positions and indices
  - Supports ALLOW_COMPACTION, PREFER_FAST_TRACE, MINIMIZE_MEMORY, ALLOW_UPDATE flags
  - Compacts BLAS post-build for memory efficiency
  - Serializes compacted BLAS for PAK archive storage
  - Skinned meshes get ALLOW_UPDATE flag for runtime refitting
  - @ray_tracing decorator (projected S10) prepares BLAS config
- **Dependencies**: T-AS-1.2, S14 RHI BLAS API, S10 ray tracing abstractions
- **Estimated Effort**: MEDIUM (3-4 days)

### T-AS-1.6 -- LOD Generation Engine
- **Description**: Implement 4 LOD strategies (Discrete, Continuous/HQE, Hierarchical, Nanite-style DAG), LOD cross-fading, and @lod decorator integration.
- **Acceptance Criteria**:
  - Discrete LOD: N independent meshes, distance-switched via @lod distances array
  - Continuous LOD: Garland-Heckbert quadric error metric simplification
  - Hierarchical LOD: Tree of simplified parents covering children's spatial region
  - Nanite-style DAG: Meshlet cluster hierarchy (research, post-cooking)
  - LOD cross-fade: screen-space dither (gradient noise + IGN patterns)
  - LOD bias from @lod bias parameter exposed as user graphics option
  - @lod decorator controls levels, distances, bias
  - All strategies idempotent
- **Dependencies**: T-AS-1.4 (meshlets for hierarchical/Nanite), S3 Materials (cross-fade materialization)
- **Estimated Effort**: HIGH (5-7 days)

### T-AS-1.7 -- Draco-glTF Decompression (KHR_draco_mesh_compression)
- **Description**: Add Draco mesh decompression as an import extension for glTF files using KHR_draco_mesh_compression.
- **Acceptance Criteria**:
  - Detects KHR_draco_mesh_compression in glTF extensionsRequired
  - Decompresses Draco-compressed geometry data during import
  - Produces identical vertex data to uncompressed equivalent
  - Integration with the existing glTF parser pipeline
  - Falls back gracefully if Draco extension not available
- **Dependencies**: T-AS-1.1, draco crate or draco WASM decoder
- **Estimated Effort**: SMALL (1-2 days)

---

## Phase 2: Texture Pipeline (7 tasks)

### T-AS-2.1 -- Base Texture Importer (stb_image Integration)
- **Description**: Implement the core texture import pipeline using stb_image: decode PNG/JPEG/TGA/BMP, format selection, pixel format mapping, memory tracking.
- **Acceptance Criteria**:
  - Decodes PNG, JPEG, TGA, BMP via stb_image
  - Maps source pixel formats to GPU formats (R8G8B8A8_UNORM, R8G8B8A8_SRGB, R16G16B16A16_FLOAT)
  - Handles sRGB detection and gamma-corrected pipeline
  - Memory budget tracking per-texture
  - @asset decorator maps .png/.jpg/.tga to stb_image loader
  - Texture state tracking: PENDING -> UPLOADING -> READY -> EVICTED
- **Dependencies**: stb_image crate, S14 RHI image creation
- **Estimated Effort**: SMALL (1-2 days)

### T-AS-2.2 -- HDR and Advanced Format Support
- **Description**: Add EXR (OpenEXR), HDR (Radiance RGBE), TIFF, PSD format decoding.
- **Acceptance Criteria**:
  - Decodes EXR (16-bit/32-bit float, lossless and lossy modes)
  - Decodes Radiance RGBE (.hdr)
  - Decodes TIFF (uncompressed)
  - Decodes PSD
  - Tone-mapping preview for HDR formats
  - Format mapping to GPU HDR formats (R16G16B16A16_FLOAT, RGB10_A2_UNORM)
- **Dependencies**: T-AS-2.1, image crate (Rust)
- **Estimated Effort**: MEDIUM (2-3 days)

### T-AS-2.3 -- KTX2 / Basis Universal Decoder
- **Description**: Implement KTX2 container parser with Basis Universal supercompressed texture decoding.
- **Acceptance Criteria**:
  - Parses KTX2 container format (header, metadata, mip levels, array layers)
  - Decodes Basis Universal supercompressed data to GPU formats
  - Supports UASTC (high quality) and ETC1S (low quality) Basis modes
  - Transcodes to BC1-7, ASTC, ETC2, and R8G8B8A8_UNORM as needed
  - Supports cubemap and texture array KTX2 files
  - @asset decorator maps .ktx2 to KTX2/Basis loader
  - GPU upload path for transcoded textures
- **Dependencies**: T-AS-2.1, basis-universal crate or KTX-Software library, S14 RHI transfer queue
- **Estimated Effort**: HIGH (5-7 days)

### T-AS-2.4 -- Mipmap Generation and Block Compression
- **Description**: Implement mip chain generation (Lanczos default), BCn/ASTC/ETC2 block compression, and @cook decorator integration.
- **Acceptance Criteria**:
  - Generates mip chains with Lanczos filter (configurable per-texture)
  - Block compression: BC1-BC7 for desktop, ASTC for mobile, ETC2 for legacy mobile
  - @cook decorator's compression parameter selects target format
  - min_mip from @residency decorator controls minimum resident mip
  - Cook-time operation, produces GPU-ready compressed data
  - Performance: <500ms per 4K texture for BC7
- **Dependencies**: T-AS-2.1, texture-compression crate (or bcn/astc-encoder crates)
- **Estimated Effort**: HIGH (5-7 days)

### T-AS-2.5 -- Cubemap and Texture Array Assembly
- **Description**: Implement cubemap assembly (cross layout, 6 individual images, KTX), texture array construction, and seam-aware mip filtering.
- **Acceptance Criteria**:
  - Detects cross/cube layout images via aspect ratio
  - Assembles 6 individual face images into cubemap
  - KTX files natively parsed as cubemap array layers
  - GPU creation: VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT / CUBEMAP
  - Seam-aware mip filtering across cubemap faces
  - Texture arrays: identical format/size/mip, 256-2048 layers
  - WGSL sampling: textureSample(array, sampler, coords, layer)
- **Dependencies**: T-AS-2.1, S14 RHI cubemap/texture array creation
- **Estimated Effort**: MEDIUM (2-3 days)

### T-AS-2.6 -- Virtual Texturing Page Extraction
- **Description**: Implement virtual texturing page extraction for S12: page-aligned tiling, mip chains in page chunks, page table init data, streaming-friendly PAK layout.
- **Acceptance Criteria**:
  - Extracts 128x128 pixel pages from source textures at all mip levels
  - Organizes mip chains in page-sized chunks
  - Produces page table initialization data
  - Stores page data in PAK archive with spatial locality
  - Page content hashes computed for ContentStore deduplication
  - Cook-time operation
- **Dependencies**: T-AS-2.4, S12 Virtual Texturing interface spec
- **Estimated Effort**: MEDIUM (3-4 days)

### T-AS-2.7 -- KTX/DDS Container Format Parsers
- **Description**: Implement KTX (v1) and DDS container parsers for pre-compressed textures with hardware formats.
- **Acceptance Criteria**:
  - Parses KTX v1 container (header, mip levels, face/array indices)
  - Parses DDS (DirectDraw Surface) with DX10 header extension
  - Detects pre-compressed BCn/ASTC formats from container metadata
  - Passes compressed data directly to GPU (no decode/re-encode)
  - @asset decorator maps .ktx and .dds to respective parsers
- **Dependencies**: T-AS-2.1, S14 RHI compressed texture support
- **Estimated Effort**: SMALL (1-2 days)

---

## Phase 3: Shader Compilation Pipeline (6 tasks)

### T-AS-3.1 -- WGSL Preprocessor Implementation
- **Description**: Implement the WGSL preprocessor: #define, #include, #ifdef, #error, predefined macros, and include path resolution.
- **Acceptance Criteria**:
  - Supports #define NAME [value], #undef NAME
  - Supports #if / #elif / #else / #endif conditionals
  - Supports #ifdef / #ifndef
  - Supports #include "path" with search path resolution
  - Supports #warning and #error directives
  - Predefined macros: TRINITY_VERSION, TRINITY_RHI_VULKAN/_D3D12/_METAL, TRINITY_MATERIAL_VARIANT_*, etc.
  - Preprocessor state serializable for deterministic recompilation
  - Dependency extraction: all #include paths recorded for hot-reload
- **Dependencies**: naga crate (for WGSL tokenization)
- **Estimated Effort**: MEDIUM (2-3 days)

### T-AS-3.2 -- naga Compilation Pipeline
- **Description**: Implement the WGSL compilation pipeline using naga: parse, validate, analyze, transpile to SPIR-V/DXIL/MSL.
- **Acceptance Criteria**:
  - Parse: WGSL source -> naga IR module with source location spans
  - Validate: type checking, binding overlap, entry point validity, spec compliance
  - Analyze: resource bindings, push constants, specialization constants, entry points
  - Transpile: SPIR-V via naga SPIR-V backend, MSL via naga Metal backend
  - DXIL path: SPIR-V -> DXC (or SPIRV-Cross for HLSL)
  - Performance: <50ms per entry point (excluding optimization)
  - Error reporting with source location spans
- **Dependencies**: naga crate, spirv-opt (optional), DXC (optional)
- **Estimated Effort**: MEDIUM (3-4 days)

### T-AS-3.3 -- Shader Reflection Engine
- **Description**: Implement shader reflection: extract bindings, push constants, specialization constants, and produce pipeline layout descriptions.
- **Acceptance Criteria**:
  - Discovers: uniform buffers, storage buffers, sampled textures, samplers, storage textures, push constants, specialization constants
  - Extracts: binding slot, size, member layout, read/write flags, dimension, array size, format
  - Drives: VkPipelineLayout / Root Signature / MTLRenderPipelineDescriptor creation
  - Drives: descriptor set allocation and update
  - Drives: push constant range specification
  - Drives: specialization constant overrides at PSO creation time
  - Performance: <10ms per entry point
- **Dependencies**: T-AS-3.2, naga IR analysis
- **Estimated Effort**: MEDIUM (2-3 days)

### T-AS-3.4 -- 3-Level Shader Cache
- **Description**: Implement the shader cache: in-memory LRU (512 MB), disk cache via ContentStore, PAK archive pre-compiled shaders, cache invalidation.
- **Acceptance Criteria**:
  - In-Memory LRU: stores full PSO or compiled bytecode, 512 MB configurable
  - Disk Cache: ContentStore-backed, keyed by content hash of (source + defines + platform + compiler version + optimization + include hashes)
  - PAK Archive: pre-compiled common shaders shipped with application
  - Cache key includes: source content hash, #define flags, target platform, language version, compiler version, optimization level, include content hashes
  - Invalidation triggers: source modification, compiler version bump, platform target change, explicit clear
  - Performance: <1ms memory lookup, <10ms disk lookup
- **Dependencies**: T-AS-3.2, Phase 4 ContentStore, S14 RHI pipeline cache
- **Estimated Effort**: MEDIUM (3-4 days)

### T-AS-3.5 -- Shader Dependency Extraction
- **Description**: Extract complete dependency graph for each shader: direct/transitive includes, @import modules, material generator DSL files.
- **Acceptance Criteria**:
  - Extracts all #include paths from preprocessed WGSL source
  - Extracts transitive #include paths (included files that include others)
  - Extracts WGSL @import module references
  - Records material generator DSL dependencies (S3 .py files)
  - Resolves all paths through configurable search paths (mount points)
  - Computes content hash of each dependency
  - Stores full dependency tree alongside compiled output
  - Dependency change: invalidates all transitively dependent shaders
- **Dependencies**: T-AS-3.1, T-AS-3.2
- **Estimated Effort**: MEDIUM (2-3 days)

### T-AS-3.6 -- Shader Edit-and-Continue Pipeline
- **Description**: Implement the runtime shader recompilation path: content hash change detection, background recompile, PSO swap at phase boundary, fence tracking.
- **Acceptance Criteria**:
  - Detects source shader change via content hash mismatch
  - Recompiles in background thread (doesn't block rendering)
  - Creates new PSO on render thread while retaining old PSO
  - Atomically swaps PSO at next RENDER phase boundary (before draw submission)
  - Retains old PSO until all in-flight frames referencing it complete (fence tracking)
  - On recompilation failure: old shader remains active, error logged
  - Editor receives notification: success or error with source location
- **Dependencies**: T-AS-3.4, T-AS-3.5, S14 RHI pipeline barriers, Phase 6 hot-reload
- **Estimated Effort**: MEDIUM (3-4 days)

---

## Phase 4: Content-Addressable Store (8 tasks)

### T-AS-4.1 -- BLAKE3 Hashing Migration
- **Description**: Replace SHA-256 with BLAKE3 for all content hashing. Implement backward-compatible migration path with dual-hash storage.
- **Acceptance Criteria**:
  - BLAKE3 implementation: SIMD-parallel hashing for large assets (10x+ faster than SHA-256)
  - Streaming incremental hashing for large file imports
  - Keyed hashing mode for authenticated asset content
  - Extendable output: configurable hash length (256, 512 bits)
  - Dual-hash mode during migration: both SHA-256 and BLAKE3 stored
  - Hash function recorded in asset manifest for backward compatibility
  - ContentHash wrapper class supports multiple hash functions
  - Performance: <5ms per 1MB data (target: PCIe 4.0 NVMe throughput)
- **Dependencies**: BLAKE3 crate, existing ContentStore ContentHash type
- **Estimated Effort**: MEDIUM (2-3 days)

### T-AS-4.2 -- ContentStore Streaming API
- **Description**: Add put_stream/get_stream to the StorageBackend protocol for efficient large file handling without loading entire files into memory.
- **Acceptance Criteria**:
  - put_stream(reader: Read) -> ContentHash: streams data through hash computation
  - get_stream(hash: ContentHash) -> Reader: returns a reader for partial data access
  - Streams use bounded buffers (configurable, default 64KB)
  - FileBackend: streams directly to/from disk without full-RAM buffer
  - MemoryBackend: accumulates to bytes (no streaming benefit, but API compatible)
  - Supports seeking within streams for partial asset reads
- **Dependencies**: T-AS-4.1, S15 core memory allocators
- **Estimated Effort**: MEDIUM (2-3 days)

### T-AS-4.3 -- ContentStore TTL/Eviction and LRU
- **Description**: Add LRU eviction with configurable budget to MemoryBackend for runtime cache management.
- **Acceptance Criteria**:
  - LRU eviction policy with configurable max size (bytes) and max entry count
  - Eviction order: least-recently-used first
  - Budget enforcement: put() that exceeds budget triggers eviction
  - TTL support: entries expire after configurable duration
  - Thread-safe: RwLock-guarded eviction
  - Notifications: eviction events published to Tracker
- **Dependencies**: T-AS-4.2, Foundation Tracker
- **Estimated Effort**: SMALL (1-2 days)

### T-AS-4.4 -- SQLite Metadata Backend
- **Description**: Implement SQLite-backed ContentStore backend for metadata queries by asset type, date, provenance, and content hash.
- **Acceptance Criteria**:
  - Stores: content hash, asset type, import date, provenance data, dependencies
  - Queries: find_by_type(asset_type), find_by_date(range), find_by_provenance(key, value)
  - Indexed on: content_hash, asset_type, import_date
  - Optional backend -- core ContentStore works without it
  - Graceful fallback to O(n) scan when SQLite not available
  - Thread-safe connection pooling
- **Dependencies**: SQLite crate, T-AS-4.2
- **Estimated Effort**: MEDIUM (2-3 days)

### T-AS-4.5 -- FileBackend Multi-Level Sharding
- **Description**: Implement multi-level directory sharding (2/4/6 character prefix) for FileBackend to handle 10k+ assets without performance degradation.
- **Acceptance Criteria**:
  - Supports 2-level (hash[:2]/hash[2:]), 4-level (hash[:4]/hash[4:]), 6-level sharding
  - Configurable sharding depth per FileBackend instance
  - Sharding migration tool: move existing 2-level store to deeper sharding
  - Performance: 10k asset lookup in <5ms per get() call
  - Backward compatible: reads from old 2-level sharding during migration
- **Dependencies**: T-AS-4.2, FileBackend (existing)
- **Estimated Effort**: SMALL (1-2 days)

### T-AS-4.6 -- CRC-32C Integrity Verification
- **Description**: Add CRC-32C checksum verification on FileBackend reads to detect corruption. Add optional replication.
- **Acceptance Criteria**:
  - CRC-32C computed on put(), stored alongside content hash
  - CRC-32C verified on every get() read
  - Corruption detected -> error reported with asset identity
  - Optional replication: N copies of each asset in FileBackend
  - Replication read: if primary copy CRC fails, try replica
  - Configurable per FileBackend instance
- **Dependencies**: T-AS-4.2, CRC-32C hardware acceleration (SSE 4.2 / ARM CRC)
- **Estimated Effort**: SMALL (1-2 days)

### T-AS-4.7 -- Provenance Chain System
- **Description**: Implement complete asset provenance tracking: source hash, import config, processing stages, cook config, dependency references, stored as tree in ContentStore.
- **Acceptance Criteria**:
  - Provenance record per asset: source hash, import timestamp + tool version, processing parameters, intermediate hashes, cook config + platform, dependency references
  - Stored in ContentStore as provenance/<asset_guid>/ tree
  - Structured as tree nodes with sentinel markers
  - Supports incremental rebuild: if source hash unchanged and params unchanged, skip processing
  - Supports reproducibility: same source hash + same params -> same output
  - Supports debugging: identify which processing step introduced artifact
  - ContentDiffer compatible for provenance tree diffing
- **Dependencies**: T-AS-4.2, Foundation ContentStore tree storage
- **Estimated Effort**: MEDIUM (2-3 days)

### T-AS-4.8 -- DeltaSync: Incremental Asset Sync (TRINITY-UNIQUE)
- **Description**: Implement DeltaSync on ContentDiffer: remote build farm sync, multi-platform cooking sharing, runtime update diffs. This is the TRINITY-UNIQUE differentiator.
- **Acceptance Criteria**:
  - Remote build farm: cooked asset diff computed against local cache, only changed content hashes transferred
  - Multi-platform cooking: identical intermediate data shared, only platform-specific output transmitted
  - Runtime updates: game client receives content hash diffs, not full asset downloads
  - ContentDiffer produces O(differences) proofs for all three use cases
  - Network transport: minimal protocol (content hash list + diffs)
  - Conflict resolution: latest timestamp wins for same content hash
  - Integration with ContentStore put()/get() for all delta operations
- **Dependencies**: T-AS-4.7, Foundation ContentDiffer, networking layer (S15)
- **Estimated Effort**: HIGH (5-7 days)

---

## Phase 5: Streaming System (7 tasks)

### T-AS-5.1 -- 3-Thread Streaming Architecture
- **Description**: Implement the background streaming thread with lock-free SPSC/MPSC queues for inter-thread communication.
- **Acceptance Criteria**:
  - Dedicated streaming thread, separate from game and render threads
  - Lock-free SPSC queue for request submission (game -> streaming)
  - Lock-free MPSC queue for priority updates (game -> streaming)
  - Ring buffer for GPU upload commands (streaming -> render)
  - Atomic state flags for asset loading stage transitions
  - Thread-safe reference counting (atomic operations)
  - Clean shutdown: drain queues, complete in-flight I/O, join thread
- **Dependencies**: S15 task_system.rs, crossbeam crate (SPSC/MPSC)
- **Estimated Effort**: MEDIUM (3-4 days)

### T-AS-5.2 -- Weighted Priority Queue
- **Description**: Implement the streaming priority queue with 5-component weighted score, binary heap, priority tiers, and lock-free skip list for contention reduction.
- **Acceptance Criteria**:
  - Priority = (visibility_weight * visibility_factor) + (velocity_weight * velocity_factor) + (distance_weight * distance_factor) + (lod_bias * lod_factor) + base_priority
  - Binary heap: O(log n) insert, update, pop
  - Priority tiers: CRITICAL (bypass), HIGH (1-2 frames), NORMAL (opportunistic), LOW (idle)
  - Lock-free skip list for contention-free priority updates
  - Cancel: mark-and-skip (not immediate removal)
  - Priority update on camera movement triggers re-heapification
  - All weights configurable via @loading_priority decorator
- **Dependencies**: T-AS-5.1, S15 task_system.rs
- **Estimated Effort**: MEDIUM (3-4 days)

### T-AS-5.3 -- Budget System with Eviction
- **Description**: Implement the GPU memory budget system: per-type budgets, global cap, LRU + cost-benefit eviction, @unloadable integration.
- **Acceptance Criteria**:
  - Budgets: Mesh 512 MB, Texture 1 GB, Shader 256 MB, Global 2 GB (all configurable)
  - I/O bandwidth budget: max bytes per frame, dynamic adjustment based on frame time
  - Pre-load estimation: compute asset footprint before loading
  - Eviction policy: LRU with cost-benefit (low-priority, large-footprint first)
  - @unloadable min_age constraint respected (assets below min age not evicted)
  - @unloadable save_state=True serializes state before GPU memory freed
  - Eviction candidates selected from pool of unpinned assets
- **Dependencies**: T-AS-5.1, Foundation Tracker (state change notifications)
- **Estimated Effort**: MEDIUM (3-4 days)

### T-AS-5.4 -- Predictive Pre-Loading
- **Description**: Implement velocity-based look-ahead predictive loading using @loading_priority heuristics.
- **Acceptance Criteria**:
  - Projects camera position at t+1s, t+2s, t+5s
  - Computes asset proximity from projected frustum
  - Boosts priority for assets in projected frustum (closer in time = higher boost)
  - @loading_priority's player_velocity_weight controls sensitivity
  - Configurable look-ahead distances
  - Integrates with priority queue (priority updates for predicted assets)
- **Dependencies**: T-AS-5.2, S15 math.rs (frustum, projection)
- **Estimated Effort**: MEDIUM (2-3 days)

### T-AS-5.5 -- Budget-Aware LOD Selection
- **Description**: Implement LOD budget: limit total loaded texels/triangles across visible assets, integrate with @lod decorator.
- **Acceptance Criteria**:
  - Texel budget: limit total texels across all loaded textures
  - Triangle budget: limit total triangles across all loaded meshes
  - LOD selection considers budget: if budget exceeded, select lower LOD
  - Priority-weighted LOD: low-priority assets reduce LOD first
  - Integration with @lod distances array and bias parameter
  - Dynamic adjustment as camera moves and budget pressure changes
- **Dependencies**: T-AS-5.3, T-AS-1.6 (LOD system)
- **Estimated Effort**: MEDIUM (2-3 days)

### T-AS-5.6 -- Remote Asset Caching
- **Description**: Implement remote asset cache: network ContentStore backend for build farm and team-wide asset deduplication.
- **Acceptance Criteria**:
  - Network-aware ContentStore backend (HTTP/REST or custom protocol)
  - Cache hit: download content hash from remote
  - Cache miss: compute locally, upload to remote
  - Transparent fallback: if remote unavailable, use local-only
  - Configurable remote endpoint and authentication
  - Integration with DeltaSync for differential updates
- **Dependencies**: T-AS-5.1, T-AS-4.8, networking layer
- **Estimated Effort**: MEDIUM (2-3 days)

### T-AS-5.7 -- Job System Integration for Decompress/Deserialize
- **Description**: Delegate decompression (LZ4/Zstd) and deserialization to the shared job system (S15 task_system.rs) with priority inheritance.
- **Acceptance Criteria**:
  - Decompression jobs submitted to task system with streaming thread priority
  - Deserialization jobs submitted to task system
  - Priority inheritance: streaming thread priority propagated to its jobs
  - I/O + decompress pipeline: overlap reading and decompressing
  - Thread-safe job completion callbacks
  - Graceful handling of job system saturation
- **Dependencies**: T-AS-5.1, S15 task_system.rs, LZ4/Zstd crate
- **Estimated Effort**: MEDIUM (2-3 days)

---

## Phase 6: Hot-Reload Infrastructure (5 tasks)

### T-AS-6.1 -- Cross-Platform File Watcher
- **Description**: Implement the file watcher with OS-native events (inotify/FSEvents/RDC), polling fallback, exclusion filters, and debounce window.
- **Acceptance Criteria**:
  - Linux: inotify-based file watching
  - macOS: FSEvents-based file watching
  - Windows: ReadDirectoryChangesW-based file watching
  - Fallback: 2-second periodic polling on unsupported systems
  - Exclusion filters: .git, .svn, *.tmp, *.autosave, build output
  - Debounce: 500ms coalescing window for rapid saves
  - Change classification: modified, created, deleted, renamed
  - Bounded event buffer: ring buffer prevents memory exhaustion
- **Dependencies**: notify crate (Rust), S15 task_system.rs
- **Estimated Effort**: MEDIUM (2-3 days)

### T-AS-6.2 -- Content-Level Change Detection
- **Description**: Implement content hash-based change detection for timestamp-ambiguous changes (git checkout), editor startup scan, and periodic integrity verification.
- **Acceptance Criteria**:
  - Content hash comparison: detect changes even when timestamps match
  - Editor startup scan: detect changes made while editor was closed
  - Optional periodic full hash scan for content integrity
  - Integration with file watcher: hash verification after filesystem event
  - Asset cache invalidation on content hash mismatch
  - Deleted file handling: mark asset as missing, log warning
- **Dependencies**: T-AS-6.1, T-AS-4.1 (BLAKE3 for hash comparison)
- **Estimated Effort**: SMALL (1-2 days)

### T-AS-6.3 -- Shader Hot-Reload Propagation
- **Description**: Implement the full shader hot-reload propagation path: dependency tree resolution, background recompile, PSO atomic swap at phase boundary, fence tracking, debug feedback.
- **Acceptance Criteria**:
  - Dependency resolution: when include file changes, find all affected entry points
  - Background recompile: compile only leaf (entry-point) shaders, not internal deps
  - Old PSO retained until new PSO is ready (no rendering stall)
  - Atomic PSO swap at RENDER phase 1 boundary (before draw submission)
  - Fence tracking: old PSO freed when all in-flight frames complete
  - On failure: old asset remains active, error logged with source location
  - Subsequent saves trigger recompilation (iterative fix-save-compile)
  - Debug feedback: success notification or error with source span to editor
- **Dependencies**: T-AS-6.2, T-AS-3.6 (shader edit-and-continue), S14 RHI pipeline barriers
- **Estimated Effort**: MEDIUM (3-4 days)

### T-AS-6.4 -- Texture and Mesh Hot-Reload Propagation
- **Description**: Implement texture hot-reload (reimport -> GPU upload -> descriptor update) and mesh hot-reload (reimport -> buffer update -> BLAS rebuild -> render proxy swap).
- **Acceptance Criteria**:
  - Texture: detect change -> reimport -> reprocess (mips, compression) -> new GPU image -> descriptor update -> optional 1-2 frame fade transition
  - Mesh: detect change -> reimport -> regenerage meshlets -> rebuild BLAS -> buffer update via staging buffer -> atomic render proxy update at phase boundary -> BLAS swap (may take multiple frames)
  - Old resource retained until new one ready
  - No rendering stalls during hot-reload
- **Dependencies**: T-AS-6.2, T-AS-1.6 (mesh LOD), T-AS-2.4 (texture compression), S14 RHI
- **Estimated Effort**: MEDIUM (3-4 days)

### T-AS-6.5 -- Material Instance Hot-Reload and Dependency Viewer
- **Description**: Implement material instance parameter hot-reload (uniform buffer push for parameter changes) and live asset dependency graph visualization.
- **Acceptance Criteria**:
  - Material parameter changes: push updated uniform buffer without full PSO recreation
  - Material DSL file change: detect via file watcher, trigger WGSL regeneration via S3, follow shader hot-reload path
  - Dependency graph visualization: Graphviz or web-based from ContentStore provenance data
  - Graph shows: asset -> dependencies -> dependents -> change propagation path
  - Real-time updates: graph refreshes when dependencies change
  - Integration with S18 Editor for dependency panel
- **Dependencies**: T-AS-6.3, Phase 4 provenance data, S3 Materials for DSL regeneration
- **Estimated Effort**: MEDIUM (2-3 days)
