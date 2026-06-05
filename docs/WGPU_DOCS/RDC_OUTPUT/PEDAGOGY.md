# PEDAGOGY — Concept Evolution Log

**Purpose:** Archaeological record of concept changes during RDC consolidation
**Update Mode:** Append-only
**Started:** 2026-05-27

---

## Format

Each entry records:
- **Timestamp:** SCRIBE pass number + source document
- **Concept:** Name/path of the concept that changed
- **Prior Value:** What MASTER said before this pass
- **New Value:** What MASTER says after this pass
- **Reason:** Why the change was made (from source doc context)
- **Court Reference:** Optional — INPROGRESS.md anchor if court-resolved

---

## Evolution Log

*Entries will be appended below as SCRIBE passes execute...*

---

## Pass 2-4: Parts I, II, III — 2026-05-27

### Entry: Chapter 1 - The wgpu Object Model (Part I)

**Concept:** Chapter 1: The wgpu Object Model (Instance, Adapter, Device, Queue)
**Prior Value:** Skeletal bullet points (~35 lines)
- 1.1.1 Instance Creation: "Entry point to wgpu; selects backend(s)"
- 1.2.3 Limits: "Max texture dimensions, buffer size, bind groups"
- 1.3.4 Device Lost: "Handling and recovery"
- (etc. - single-line descriptions)

**New Value:** Comprehensive implementation guide (600+ lines)
- Full Instance creation with InstanceDescriptor, backend selection table (6 platforms)
- Backends bitflags with PRIMARY/SECONDARY aliases
- InstanceFlags struct with performance impact table (5 flags)
- TrinityGraphics struct with graceful fallback adapter selection
- AdapterInfo, DeviceType enums, vendor ID table (6 vendors)
- Complete Limits struct (~25 fields with typical values)
- Features bitflags (~30 features), FeatureTier enum (5 tiers)
- AdapterSelector with scoring, blacklisting, vendor preference
- DeviceRequirements struct with required/optional features per tier
- DeviceManager with lost callback, recovery strategy
- ErrorScope RAII wrapper
- TrinityDevice with atomic tracking, frame management
- QueueSync with on_submitted_work_done callback
- SubmissionBatcher with size/time thresholds
- Device module architecture (8 files)

**Reason:** WGPU_PART_I_DEVICE_INSTANCE.md provides the complete device initialization lifecycle from Instance creation through Device recovery, essential for robust engine startup.

---

### Entry: Chapter 2 - Buffers (Part II)

**Concept:** Chapter 2: Buffers
**Prior Value:** Skeletal bullet points (~15 lines)
- "Usage Flags: VERTEX, INDEX, UNIFORM, STORAGE, INDIRECT, ..."
- "Mapping: Synchronous vs asynchronous"
- "Vertex buffers, Index buffers (u16/u32)"

**New Value:** Comprehensive buffer management (350+ lines)
- BufferDescriptor fields table (4 fields)
- BufferUsages bitflags (10 flags) with usage combinations table (6 patterns)
- Sync mapping with mapped_at_creation, async mapping with map_async
- TRINITY async mapping wrapper using oneshot channel
- DeferredDestroyer with frame-based deferred deletion
- Vertex struct with vertex_attr_array macro
- VertexFormatRegistry with standard_pbr (48 bytes) and skinned (72 bytes)
- Dynamic uniform buffers with UNIFORM_ALIGNMENT (256)
- DrawIndexedIndirectArgs, DispatchIndirectArgs struct definitions
- RingBuffer with wrap-around logic and frame tracking
- TrinityBufferSystem with triple-buffering

**Reason:** WGPU_PART_II_RESOURCES.md's buffer chapter covers all buffer types from staging to indirect, with TRINITY-specific memory management patterns.

---

### Entry: Chapter 3 - Textures (Part II)

**Concept:** Chapter 3: Textures
**Prior Value:** Skeletal bullet points (~25 lines)
- "Dimensions: 1D, 2D, 3D"
- "Formats: Color, depth, stencil, compressed"
- "Usage: TEXTURE_BINDING, STORAGE_BINDING, RENDER_ATTACHMENT"

**New Value:** Comprehensive texture management (250+ lines)
- TextureDescriptor with view_formats for reinterpretation
- Dimension interpretation table (D1/D2/D3 x width/height/depth)
- Color format table (7 common formats with bytes and description)
- Depth format table (4 formats)
- Compressed format table (4 BC formats with block size and ratio)
- TextureUsages bitflags (5 flags)
- calculate_mip_count function, MipGenerator compute shader loop
- Cube map creation with TextureViewDimension::Cube
- TextureFormatSelector helpers: color_attachment, depth, normal_map
- Texture views: mip-level, layer, aspect, format reinterpretation
- SamplerDescriptor with all fields, shadow sampler configuration
- SamplerCache with 4 presets (linear_repeat, linear_clamp, point, shadow)
- Texture operations: write_texture, copy commands

**Reason:** WGPU_PART_II_RESOURCES.md's texture chapter provides format selection strategy and sampler caching essential for material systems.

---

### Entry: Chapter 4 - Bind Groups & Layouts (Part II)

**Concept:** Chapter 4: Bind Groups & Layouts
**Prior Value:** Skeletal bullet points (~20 lines)
- "Bind group concept (descriptor set equivalent)"
- "Bindless via storage buffers"
- "Push constants (platform-dependent)"

**New Value:** Comprehensive binding model (150+ lines)
- Full BindGroupLayoutDescriptor example (3 bindings: uniform, texture, sampler)
- BindGroupDescriptor with matching entries
- BindingType::Buffer storage configuration
- Texture array layout with count for bindless
- BindlessManager with allocation, free, index recycling via free_indices
- PipelineLayoutDescriptor with 4 bind groups, 2 push constant ranges
- Resources module architecture (8 files)

**Reason:** WGPU_PART_II_RESOURCES.md's bind group chapter enables material diversity through bindless architecture.

---

### Entry: Chapter 5 - WGSL & Naga (Part III)

**Concept:** Chapter 5: WGSL & Naga
**Prior Value:** Skeletal bullet points (~25 lines)
- "Types: Scalar, vector, matrix, array, struct"
- "Pipeline: WGSL -> IR -> Validation -> Backend"
- "Override constants (pipeline-overridable)"

**New Value:** Comprehensive shader compilation guide (500+ lines)
- Full vertex/fragment shader example with CameraUniform
- WGSL types: scalar (5), vector (4), matrix (4), array, struct with swizzling
- Address spaces: function, private, workgroup, uniform, storage, handle
- Built-in functions: math (~20), geometric (~8), texture (~12), derivative (6), atomic (10), sync (2), pack/unpack (10)
- Attributes: entry point (3), binding (4), interpolation (3), other (4)
- Built-in variables: vertex (3), fragment (5), compute (5)
- Naga architecture diagram (Frontend->IR->Validation->Backend)
- WGSL frontend: parse_str, Validator with flags/capabilities
- Backend targets: SPIR-V, MSL, HLSL, GLSL options structs
- ShaderCache with in_memory HashMap + disk cache
- ShaderHotReload with notify watcher, pending_reloads channel
- Shader module creation: WGSL and SPIR-V patterns
- Naga pre-validation for better error messages
- ShaderReflection from naga Module
- Override constants: @id, PipelineCompilationOptions
- ShaderPermutationManager with FeatureFlags bitflags (7 flags)
- ShaderVariantSystem with precompile_common_variants
- Shader directory structure (common, vertex, fragment, compute, rt)

**Reason:** WGPU_PART_III_SHADERS.md provides the complete WGSL reference and Naga compilation pipeline, enabling shader variant systems.

---

## Pass 5-7: Parts VII, VIII, IX — 2026-05-27

### Entry: Chapter 13 - RT Pipeline (Part VII)

**Concept:** Chapter 13: Ray Tracing Pipelines  
**Prior Value:** Skeletal bullet points (14 lines)
- `ray_tracing_pipeline` feature (experimental)
- Stages: Ray generation, intersection, any-hit, closest-hit, miss, callable
- Recursion depth
- (etc. - simple lists)

**New Value:** Comprehensive implementation guide (450+ lines)
- Full shader table dispatch model diagram
- Pipeline vs Ray Query comparison table (7 dimensions)
- All 6 shader stages with WGSL examples
- Complete SBT builder implementation
- RT pipeline creation API (speculative)
- 5 RT patterns: primary, shadow, reflection, GI, path tracing
- TRINITY implementation roadmap

**Reason:** WGPU_PART_VII_RT_PIPELINE.md provides the most detailed wgpu RT documentation. Code examples are speculative but based on Vulkan/DXR patterns to enable zero-latency adoption when wgpu stabilizes.

---

### Entry: Chapter 14 - RT Advanced Features (Part VII)

**Concept:** Chapter 14: RT Advanced Features  
**Prior Value:** 4 skeletal subsections (16 lines total)
- OMM: "Alpha testing acceleration, wgpu status: not yet"
- DMM: "Micro-geometry detail, wgpu status: not yet"
- SER: "Coherent ray sorting, wgpu status: not yet"
- Motion Blur: "Motion BLAS/TLAS, wgpu status: not yet"

**New Value:** Detailed feature documentation (120+ lines)
- OMM: Problem analysis, micropolygon grid diagram, building pipeline
- DMM: Concept explanation, use cases (terrain, tiled surfaces)
- SER: Coherence problem visualization, before/after thread diagrams
- Motion Blur: Motion AS structures, traceRayMotion API
- TRINITY preparation strategies for each

**Reason:** Even for future features, understanding the problem space and TRINITY preparation enables architectural readiness.

---

### Entry: Chapter 15 - Indirect Rendering (Part VIII)

**Concept:** Chapter 15: Indirect Rendering  
**Prior Value:** 10 skeletal lines
- DrawIndirect, DrawIndexedIndirect buffer layouts
- GPU-driven draw call generation
- Indirect count

**New Value:** Production-ready GPU-driven pipeline (300+ lines)
- Full struct definitions with byte sizes
- Complete WGSL compute shader for draw generation
- Frustum culling with plane extraction
- HiZ pyramid generation shader
- GPU LOD selection with distance thresholds
- Stream compaction with prefix sum
- GPUCullingPipeline struct with 5 compute stages

**Reason:** WGPU_PART_VIII_ADVANCED.md contains the complete GPU culling implementation that scales to massive scenes.

---

### Entry: Chapter 17 - Bindless Resources (Part VIII)

**Concept:** Chapter 17: Bindless Resources  
**Prior Value:** 8 skeletal lines
- Bindless texture/buffer arrays
- Descriptor indexing, non-uniform indexing
- Patterns: atlas, array, storage indirection

**New Value:** Complete bindless architecture (200+ lines)
- binding_array WGSL syntax with required features
- MaterialDescriptor storage buffer pattern
- TextureRegistry with slot allocation/recycling
- BufferRegistry with dirty range tracking
- MaterialTable combining texture + buffer registries
- IndexAllocator with free list recycling
- TRINITY summary table (9 features with status)

**Reason:** Bindless is critical for material diversity without rebinding overhead.

---

### Entry: Chapter 18 - Surface & Swapchain (Part IX)

**Concept:** Chapter 18: Surface & Swapchain  
**Prior Value:** 13 skeletal lines
- Surface creation, capabilities, formats
- Present modes, alpha modes, sRGB
- Frame acquisition, presentation

**New Value:** Complete presentation system (250+ lines)
- TrinitySurface struct with automatic config selection
- Platform format table (Windows/macOS/Linux/Web)
- Present mode comparison (VSync, tearing, latency)
- FramePacer with variance calculation
- TripleBufferConfig presets (low_latency, smooth, uncapped)
- PresentationEngine with FrameSynchronizer
- HeadlessRenderer for offline rendering
- MultiWindowRenderer with shared device

**Reason:** WGPU_PART_IX_PRESENTATION.md provides the complete frame acquisition and presentation lifecycle.

---

## Pass 8-10: Parts X, XI, XII — 2026-05-27

### Entry: Chapter 19 - Platform Considerations (Part X)

**Concept:** Chapter 19: Platform Considerations  
**Prior Value:** 12 skeletal lines
- Vulkan: Instance/device mapping, extension requirements, validation layers
- Metal: Device selection, feature sets, argument buffers
- DX12: Device selection, feature levels, root signature mapping
- WebGPU: Browser compatibility, spec conformance, WASM integration
- OpenGL: ES/WebGL fallback, limitations, performance

**New Value:** Complete multi-backend reference (500+ lines)
- Vulkan: wgpu-to-Vulkan concept mapping table (10 rows), raw handle access, 6 required device extensions, VulkanFeatures struct, validation layer env vars
- Metal: GPU family table (Apple 1-7, Mac 1-2), MetalCapabilities with family detection, unified memory handling, argument buffer bindless
- DX12: WARP filtering, feature level table (11_0-12_2), root signature mapping table (6 rows), DXC/FXC shader compiler selection
- WebGPU: Browser compatibility table (4 browsers), spec conformance validation, WebGPULimitations struct, WASM canvas integration
- OpenGL: GLES instance creation, feature comparison table (5 features x 3 backends), GLESOptimizations for mobile

**Reason:** WGPU_PART_X_PLATFORM.md documents all backend-specific behaviors essential for cross-platform optimization and debugging.

---

### Entry: Chapter 20 - Feature Detection & Capability Abstraction (Part X)

**Concept:** Chapter 20: Feature Detection & Capability Abstraction  
**Prior Value:** 8 skeletal lines
- Core/Optional/Experimental features
- Limits: texture dimensions, buffers, bind groups
- TRINITY Capability System: tiers, fallbacks

**New Value:** Complete capability abstraction system (350+ lines)
- Core feature list (8 always-available features)
- OptionalFeatures struct with 16 queryable features
- FeatureDependencies expansion logic
- inspect_limits() covering 4 limit categories (20+ limits)
- LimitRequirements for TRINITY (5 minimums)
- CapabilityTier enum (Minimal/Standard/Advanced/Full) with detection logic
- RenderPath enum with tier requirements
- CapabilityManager with automatic fallback selection
- CapabilityReport struct with 9 fields
- Platform Support Matrix (8 features x 5 backends)

**Reason:** TRINITY's tiered capability system enables graceful degradation from ray-traced to forward rendering based on hardware.

---

### Entry: Chapter 21 - Debugging (Part XI)

**Concept:** Chapter 21: Debugging  
**Prior Value:** 6 skeletal lines
- wgpu: WGPU_VALIDATION, debug markers/groups/labels, error scopes
- External: RenderDoc, PIX, Xcode, Nsight, RGP
- TRINITY: visualization modes, resource inspection, pipeline state dump

**New Value:** Complete debugging infrastructure (300+ lines)
- Validation: create_validated_instance(), validation catches list (6 items), env vars (4)
- Debug markers: DebugGroup RAII struct with push/pop
- Error scopes: with_error_scope() async pattern
- RenderDoc: RenderDocCapture struct with start/end/trigger, keyboard shortcuts
- PIX/Xcode/Nsight/RGP: Feature summaries
- DebugVisualization: 18-mode enum with shader defines
- ResourceInspector: texture readback, get_pixel() method
- PipelineStateDump: 8-field state capture
- FrameCaptureSystem: auto-capture on error/slow frame

**Reason:** WGPU_PART_XI_DEBUGGING.md provides comprehensive debug tooling from validation to external debugger integration.

---

### Entry: Chapter 22 - Profiling (Part XI)

**Concept:** Chapter 22: Profiling  
**Prior Value:** 8 skeletal lines
- GPU timing: timestamp queries, pass timing, resolution
- Pipeline statistics: invocations, clipping
- Memory: tracking, budget, leak detection
- TRINITY: per-pass timing, dashboard, bottleneck analysis

**New Value:** Complete profiling system (350+ lines)
- GPUProfiler: QuerySet creation, begin/end region, resolve, async read_results
- TimestampWriter trait for RenderPass/ComputePass
- Timer resolution methods (is_high_resolution, minimum_measurable_time)
- PipelineStatistics struct (5 u64 fields)
- overdraw_estimate(), culling_efficiency() metrics
- MemoryTracker: allocation tracking, estimate_texture_size, top_allocations
- MemoryReport struct with 6 metrics
- BudgetStatus enum (Ok/Warning/Exceeded)
- LeakDetector with frame-based detection, LeakWarning struct
- FrameProfiler with CPU/GPU regions, history averaging
- DrawCallStats with efficiency_score()
- Bottleneck enum (6 types), suggest_optimizations() returning 9 suggestions
- Debug & Profiling Tool Summary table (7 tools)

**Reason:** WGPU_PART_XI_DEBUGGING.md's profiling section enables data-driven performance optimization.

---

### Entry: Chapter 23 - Frame Graph Integration (Part XII)

**Concept:** Chapter 23: Frame Graph Integration  
**Prior Value:** 8 skeletal lines
- Resources: virtual, transient, external, aliasing
- Passes: render, compute, RT, copy
- Barriers: automatic placement, state tracking, batching
- Execution: scheduling, async compute, lifetime management

**New Value:** Complete frame graph architecture (400+ lines)
- FrameGraph struct with passes, resources, edges, execution_order
- ResourceNode with descriptor and lifetime
- ResourceDescriptor enum (Buffer/Texture)
- ResourceLifetime enum (Transient/Persistent/External/Imported)
- TransientResourcePool with acquire/release/gc
- import_texture(), import_swapchain() methods
- AliasingInfo/AliasingAnalyzer for memory sharing
- PassNode with PassType enum (Render/Compute/RayTracing/Copy)
- RenderPassConfig with attachments, LoadOp/StoreOp enums
- BarrierResolver with resource state tracking
- TextureLayout enum (8 states), needs_barrier_to() logic
- Pass scheduling with topological_sort()
- AsyncComputeScheduler with SyncDirection
- ResourceLifetimeManager with begin/end frame
- ResourceRecycler for frame-to-frame reuse

**Reason:** WGPU_PART_XII_INTEGRATION.md documents TRINITY's central rendering abstraction for automatic resource and barrier management.

---

### Entry: Chapter 24 - Python Bridge (Part XII)

**Concept:** Chapter 24: Python Bridge  
**Prior Value:** 8 skeletal lines
- PyO3: type marshalling, handle management, callbacks, errors
- Descriptors: Python-side, validation, translation, caching
- Commands: builder, deferred execution, batching, error handling

**New Value:** Complete Python bridge implementation (300+ lines)
- PyTextureDescriptor with pyo3 get/set, to_wgpu() conversion
- parse_format() mapping 9 format strings to wgpu::TextureFormat
- parse_usage() mapping 5 usage strings
- PyRenderer with Arc<Mutex<TrinityRenderer>>
- PyResourceHandle struct
- PyRenderCallback with PyObject
- wgpu_error_to_py() conversion (3 error types)
- Python TextureDesc dataclass with Enum types
- to_native() method for Rust interop
- Descriptor validation with size checks
- DescriptorCache with hash-based lookup
- RenderPassBuilder class (Python) with fluent API
- ComputePassBuilder class (Python)
- RecordedCommand enum (8 command types)
- TrinityError class hierarchy (3 subclasses)
- Complete Python API example (30 lines)
- Integration Component Summary table (6 components)

**Reason:** WGPU_PART_XII_INTEGRATION.md's Python bridge enables rapid prototyping and scripting of rendering pipelines.

---

## Pass 11-13: Parts IV, V, VI — 2026-05-27

### Entry: Chapter 6 - Graphics Pipeline (Part IV)

**Concept:** Chapter 6: Graphics Pipeline  
**Prior Value:** 14 skeletal lines
- Pipeline creation: Descriptor, layout, vertex/primitive/depth/multisample/fragment state
- Vertex input: Buffer layouts, attribute formats, step modes
- Primitive assembly: Topologies, index formats, culling
- Rasterization: Viewport, scissor, depth bias, conservative
- Fragment processing: Color targets, write masks, blending
- Depth/stencil: Depth test, compare functions, stencil operations
- Multisampling: Sample count, MSAA resolve

**New Value:** Comprehensive graphics pipeline specification (500+ lines)
- Full RenderPipelineDescriptor with all 9 fields
- Pipeline layout with bind group grouping and push constants
- Vertex state with dual buffer layout (geometry + instance)
- 32 vertex attribute formats documented in table
- Primitive state with all 7 configuration fields
- Depth/stencil state with complete face states and bias
- Fragment state with MRT (3 color targets: albedo, normal, material)
- PipelineCache with PipelineKey hash-based caching
- VertexFormatRegistry with 5 preset formats (StaticMesh, SkinnedMesh, Terrain, Particle, UI)
- All 5 primitive topologies with ASCII diagrams
- Interleaved vs separate buffer trade-offs table
- Blend modes: alpha, premultiplied, additive, multiply
- Sample count query with feature flags
- Module architecture (7 files)

**Reason:** WGPU_PART_IV_RENDER_PIPELINE.md provides the complete graphics pipeline specification from vertex input through fragment output, essential for all rendering operations.

---

### Entry: Chapter 7 - Render Passes (Part IV)

**Concept:** Chapter 7: Render Passes  
**Prior Value:** 10 skeletal lines
- Fundamentals: Color attachments, depth/stencil, queries
- Attachment operations: Load/Store, MSAA resolve
- Commands: Pipeline/buffer binding, state
- Draw commands: draw(), draw_indexed(), indirect variants
- Render bundles: Pre-recorded commands, caching

**New Value:** Complete render pass API documentation (200+ lines)
- Full RenderPassDescriptor with all attachment configurations
- LoadOp/StoreOp enums with usage patterns
- 8 set_* methods for dynamic state
- 7 draw command variants including multi-draw indirect count
- RenderBundleEncoderDescriptor with color/depth formats
- Bundle recording pattern for static geometry
- TRINITY RenderBundleCache with key-based lookup
- execute_bundles() for bundle replay

**Reason:** Render passes are the primary GPU work submission mechanism; complete API coverage enables efficient batching and state management.

---

### Entry: Chapter 8 - Compute Fundamentals (Part V)

**Concept:** Chapter 8: Compute Fundamentals  
**Prior Value:** 12 skeletal lines
- Pipeline: Descriptor, layout, entry point, caching
- Shaders: @compute, @workgroup_size, built-ins, workgroup memory, barriers
- Pass: Pipeline binding, bind groups, push constants
- Dispatch: dispatch_workgroups, indirect, limits
- Patterns: Reduction, scan, compact, sort, histogram, image, physics

**New Value:** Comprehensive compute pipeline guide (400+ lines)
- Full ComputePipelineDescriptor with compilation options
- Multiple entry points in single module
- ComputePipelineCache with specialization constants
- All 5 compute built-in variables documented
- Workgroup size considerations table (5 factors)
- 5 common workgroup size configurations
- Workgroup memory with array/atomic types
- workgroupBarrier/storageBarrier/textureBarrier documentation
- Direct dispatch with sizing patterns
- Indirect dispatch with GPU-driven count
- DispatchHelper utility struct
- 8 complete compute patterns with WGSL:
  - Parallel reduction (tree reduction)
  - Prefix scan (Blelloch algorithm)
  - Stream compaction
  - Radix sort (4-bit digit)
  - Histogram (local-to-global atomic)
  - Gaussian blur (shared tile)
  - Particle physics (Verlet integration)
- TRINITY ComputeLibrary (17 pipelines across 4 categories)
- Module architecture (9 files in 4 directories)

**Reason:** WGPU_PART_V_COMPUTE.md provides production-ready compute patterns that are the foundation for GPU-driven rendering, physics, and post-processing.

---

### Entry: Chapter 9 - Command Encoding (Part VI)

**Concept:** Chapter 9: Command Encoding  
**Prior Value:** 10 skeletal lines
- Encoder: Creation, scope, lifetime, pass creation, finalization
- Copy: Buffer-to-buffer/texture, texture-to-buffer/texture, alignment
- Clear: Buffer, texture (via pass)
- Query: Timestamp, occlusion, statistics, resolve
- Debug: Groups, markers, labels, RenderDoc/PIX/Xcode

**New Value:** Complete command encoding API (350+ lines)
- CommandEncoderDescriptor with label
- TrinityCommandEncoder with frame tracking
- Encoder scope/lifetime diagram (4 states)
- Pass encoder creation via begin_* methods
- Command buffer finalization with validation
- Copy commands with full constraint documentation
- Copy alignment table (4 constraints)
- CopyAlignmentCalculator utility
- clear_buffer with alignment requirements
- Texture clear via LoadOp::Clear in render pass
- Non-zero fill via compute shader
- TimestampQueryPool with resolve/readback pipeline
- OcclusionQuerySystem with begin/end/is_visible
- PipelineStatisticsTypes bitflags
- Query resolution pattern
- TimestampResult with duration_ns/duration_ms
- DebugScope RAII pattern for push/pop
- Debug marker insertion
- Debug label naming conventions
- RenderDocCapture with trigger methods

**Reason:** Command encoding is the core GPU work recording mechanism; complete documentation enables efficient command buffer construction and debugging.

---

### Entry: Chapter 10 - Synchronization (Part VI)

**Concept:** Chapter 10: Synchronization  
**Prior Value:** 8 skeletal lines
- Implicit: Auto barriers, usage tracking, pass ordering
- Explicit: workgroupBarrier, storageBarrier, textureBarrier
- CPU-GPU: Mapping callbacks, poll(), frame pacing
- State tracking: States, barriers, split barriers, frame graph

**New Value:** Comprehensive synchronization guide (400+ lines)
- wgpu's automatic barrier insertion examples
- Resource state table (9 buffer states, 6 texture states)
- Pass ordering semantics (sequential guarantee)
- When implicit sync is sufficient (3 criteria)
- workgroupBarrier with prefix sum example
- storageBarrier for producer-consumer pattern
- textureBarrier (experimental)
- Full memory barrier pattern
- Buffer mapping async pattern
- BufferReadback utility struct
- Maintain enum (Wait/WaitForSubmission/Poll)
- MaintainResult enum
- Async mapping with tokio
- FrameFence for submission tracking
- Single/Double/Triple buffering strategies
- TrinityFrameSynchronizer with N-buffered pacing
- Resource state table (7 states)
- BarrierType enum (RAW/WAR/WAW)
- Split barriers concept
- BarrierResolver with state tracking
- AccessFlags and PipelineStage bitflags
- needs_barrier() hazard detection
- Synchronization summary table (8 scenarios)

**Reason:** WGPU_PART_VI_SYNCHRONIZATION.md documents the most complex aspect of GPU programming; comprehensive coverage prevents data races and ensures correct multi-frame resource management.

---
