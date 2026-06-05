# WGPU_TOC.md — Complete wgpu Implementation Landscape

> **Purpose**: Dissertation-level taxonomy of the complete wgpu API surface for TRINITY
> **Scope**: Full wgpu coverage — not cherry-picked phases, but the entire graphics/compute/RT stack
> **Generated**: 2026-05-26
> **wgpu version target**: 25.x+ (with all stable + emerging features)

---

## Preface: Why This Document

The GAPSET phase model (P1/P2/P3) artificially segments wgpu into "what's available now" vs "what's gated." This creates a false impression that we're building a partial implementation.

**This document reframes the work**: TRINITY will implement the **full wgpu surface**. Phase gates are merely scheduling constraints, not architectural boundaries. The architecture must be designed for the complete picture from day one.

---

# PART I: DEVICE & INSTANCE MODEL

## Chapter 1: The wgpu Object Model

### 1.1 Instance
- 1.1.1 Instance creation and backend selection
- 1.1.2 Backend enumeration: Vulkan, Metal, DX12, OpenGL, WebGPU
- 1.1.3 Instance flags and debugging layers
- 1.1.4 TRINITY's multi-backend strategy

### 1.2 Adapter
- 1.2.1 Adapter enumeration and selection criteria
- 1.2.2 Adapter properties: vendor, device type, driver info
- 1.2.3 Adapter limits: max texture dimensions, max buffer size, max bind groups
- 1.2.4 Feature detection and capability matrices
- 1.2.5 Power preference: high-performance vs low-power
- 1.2.6 TRINITY's adapter selection algorithm

### 1.3 Device
- 1.3.1 Device creation from adapter
- 1.3.2 Required vs optional features
- 1.3.3 Device limits and limit negotiation
- 1.3.4 Device lost handling and recovery
- 1.3.5 Error scopes and error handling
- 1.3.6 TRINITY's device lifecycle management

### 1.4 Queue
- 1.4.1 Queue submission model
- 1.4.2 Command buffer submission
- 1.4.3 Queue write operations (buffer, texture)
- 1.4.4 Queue synchronization semantics
- 1.4.5 Multi-queue considerations (where supported)
- 1.4.6 TRINITY's submission batching strategy

---

# PART II: RESOURCE MODEL

## Chapter 2: Buffers

### 2.1 Buffer Fundamentals
- 2.1.1 Buffer creation and descriptor
- 2.1.2 Buffer usage flags: VERTEX, INDEX, UNIFORM, STORAGE, INDIRECT, COPY_SRC, COPY_DST, MAP_READ, MAP_WRITE, QUERY_RESOLVE
- 2.1.3 Buffer mapping: synchronous vs asynchronous
- 2.1.4 Mapped ranges and mapping modes
- 2.1.5 Buffer destruction and resource cleanup

### 2.2 Buffer Types by Role
- 2.2.1 Vertex buffers and vertex layouts
- 2.2.2 Index buffers (u16 vs u32)
- 2.2.3 Uniform buffers and dynamic offsets
- 2.2.4 Storage buffers (read-only vs read-write)
- 2.2.5 Indirect buffers (draw indirect, dispatch indirect)
- 2.2.6 Query resolve buffers
- 2.2.7 Staging buffers for upload/readback

### 2.3 Buffer Memory Management
- 2.3.1 Memory allocation strategies
- 2.3.2 Buffer suballocation and pooling
- 2.3.3 Ring buffers for per-frame data
- 2.3.4 Persistent mapping patterns
- 2.3.5 TRINITY's buffer allocator architecture

## Chapter 3: Textures

### 3.1 Texture Fundamentals
- 3.1.1 Texture creation and descriptor
- 3.1.2 Texture dimensions: 1D, 2D, 3D
- 3.1.3 Texture formats: color, depth, stencil, compressed
- 3.1.4 Texture usage flags: TEXTURE_BINDING, STORAGE_BINDING, RENDER_ATTACHMENT, COPY_SRC, COPY_DST
- 3.1.5 Mip levels and mip generation
- 3.1.6 Array layers and cube maps
- 3.1.7 Multisampled textures

### 3.2 Texture Formats Deep Dive
- 3.2.1 Uncompressed formats: R8, RG8, RGBA8, R16, RG16, RGBA16, R32, RG32, RGBA32
- 3.2.2 Float formats: R16Float, RG16Float, RGBA16Float, R32Float, RG32Float, RGBA32Float
- 3.2.3 Signed/unsigned normalized formats
- 3.2.4 sRGB formats and gamma correction
- 3.2.5 Depth formats: Depth16Unorm, Depth24Plus, Depth24PlusStencil8, Depth32Float, Depth32FloatStencil8
- 3.2.6 Compressed formats: BC (DXT), ETC2, ASTC
- 3.2.7 Format capability queries
- 3.2.8 TRINITY's format selection strategy

### 3.3 Texture Views
- 3.3.1 View creation and descriptor
- 3.3.2 View dimensions vs texture dimensions
- 3.3.3 View format reinterpretation
- 3.3.4 Mip level and array layer subranges
- 3.3.5 Aspect selection (color, depth, stencil)
- 3.3.6 View lifetime and ownership

### 3.4 Samplers
- 3.4.1 Sampler creation and descriptor
- 3.4.2 Address modes: ClampToEdge, Repeat, MirrorRepeat, ClampToBorder
- 3.4.3 Filter modes: Nearest, Linear
- 3.4.4 Mipmap filter modes
- 3.4.5 Anisotropic filtering
- 3.4.6 Comparison samplers for shadow mapping
- 3.4.7 LOD clamping
- 3.4.8 Border color
- 3.4.9 TRINITY's sampler cache

### 3.5 Texture Operations
- 3.5.1 Texture uploads via queue.write_texture
- 3.5.2 Texture copies: texture-to-texture, buffer-to-texture, texture-to-buffer
- 3.5.3 Copy layout constraints (row pitch, alignment)
- 3.5.4 Mip generation strategies
- 3.5.5 Texture streaming and virtual texturing foundations

## Chapter 4: Bind Groups & Layouts

### 4.1 Binding Model Fundamentals
- 4.1.1 The bind group concept
- 4.1.2 Bind group layouts: the contract
- 4.1.3 Bind group creation from layout
- 4.1.4 Binding types: buffer, sampler, texture, storage texture, external texture
- 4.1.5 Binding visibility: vertex, fragment, compute

### 4.2 Buffer Bindings
- 4.2.1 Uniform buffer bindings
- 4.2.2 Storage buffer bindings (read-only vs read-write)
- 4.2.3 Dynamic uniform buffers
- 4.2.4 Dynamic storage buffers
- 4.2.5 Minimum binding size constraints

### 4.3 Texture & Sampler Bindings
- 4.3.1 Sampled texture bindings
- 4.3.2 Sampler bindings
- 4.3.3 Storage texture bindings (write-only, read-write)
- 4.3.4 Multisampled texture bindings
- 4.3.5 Depth texture bindings
- 4.3.6 External texture bindings (video)

### 4.4 Advanced Binding Patterns
- 4.4.1 Bindless resources via storage buffers
- 4.4.2 Texture arrays for bindless texturing
- 4.4.3 Descriptor indexing patterns
- 4.4.4 Partially bound descriptors
- 4.4.5 TRINITY's bindless architecture

### 4.5 Pipeline Layouts
- 4.5.1 Pipeline layout creation
- 4.5.2 Bind group layout grouping strategy
- 4.5.3 Push constants (where supported)
- 4.5.4 Layout compatibility rules
- 4.5.5 TRINITY's layout caching and deduplication

---

# PART III: SHADER COMPILATION

## Chapter 5: WGSL & Naga

### 5.1 WGSL Language
- 5.1.1 WGSL syntax fundamentals
- 5.1.2 Types: scalar, vector, matrix, array, struct
- 5.1.3 Address spaces: function, private, workgroup, uniform, storage, handle
- 5.1.4 Built-in functions
- 5.1.5 Attributes: @vertex, @fragment, @compute, @binding, @group, @location, @builtin
- 5.1.6 Entry points and shader stages
- 5.1.7 WGSL extensions and feature gates

### 5.2 Naga Compiler Pipeline
- 5.2.1 Naga architecture overview
- 5.2.2 Frontend: WGSL → IR
- 5.2.3 IR representation and optimization
- 5.2.4 Validation passes
- 5.2.5 Backend targets: SPIR-V, MSL, HLSL, GLSL, WGSL
- 5.2.6 Compilation caching strategies
- 5.2.7 TRINITY's shader hot-reload system

### 5.3 Shader Modules
- 5.3.1 Shader module creation
- 5.3.2 Compilation error handling
- 5.3.3 Shader reflection (where available)
- 5.3.4 Shader module caching
- 5.3.5 Runtime shader compilation vs ahead-of-time

### 5.4 Shader Specialization
- 5.4.1 Override constants (pipeline-overridable constants)
- 5.4.2 Specialization constant patterns
- 5.4.3 Shader permutation management
- 5.4.4 TRINITY's shader variant system

---

# PART IV: RENDER PIPELINE

## Chapter 6: Graphics Pipeline

### 6.1 Pipeline Creation
- 6.1.1 Render pipeline descriptor
- 6.1.2 Pipeline layout association
- 6.1.3 Vertex state configuration
- 6.1.4 Primitive state configuration
- 6.1.5 Depth/stencil state configuration
- 6.1.6 Multisample state configuration
- 6.1.7 Fragment state and color targets
- 6.1.8 Pipeline caching and PSO management

### 6.2 Vertex Input
- 6.2.1 Vertex buffer layouts
- 6.2.2 Vertex attributes and formats
- 6.2.3 Step modes: Vertex vs Instance
- 6.2.4 Interleaved vs separate attribute buffers
- 6.2.5 TRINITY's vertex format registry

### 6.3 Primitive Assembly
- 6.3.1 Primitive topologies: PointList, LineList, LineStrip, TriangleList, TriangleStrip
- 6.3.2 Index formats: Uint16, Uint32
- 6.3.3 Front face winding
- 6.3.4 Cull modes: None, Front, Back
- 6.3.5 Polygon modes: Fill, Line (where supported)
- 6.3.6 Unclipped depth

### 6.4 Rasterization
- 6.4.1 Viewport and scissor
- 6.4.2 Depth bias (polygon offset)
- 6.4.3 Conservative rasterization (where supported)
- 6.4.4 Sample mask
- 6.4.5 Alpha to coverage

### 6.5 Fragment Processing
- 6.5.1 Fragment shader outputs
- 6.5.2 Color target state
- 6.5.3 Write mask
- 6.5.4 Blending: source factor, destination factor, operation
- 6.5.5 Blend constants
- 6.5.6 Logic operations (where supported)

### 6.6 Depth/Stencil
- 6.6.1 Depth test configuration
- 6.6.2 Depth compare functions
- 6.6.3 Depth write enable
- 6.6.4 Depth bounds (where supported)
- 6.6.5 Stencil test configuration
- 6.6.6 Stencil operations: Keep, Zero, Replace, Invert, IncrementClamp, DecrementClamp, IncrementWrap, DecrementWrap
- 6.6.7 Stencil read/write masks
- 6.6.8 Stencil reference values

### 6.7 Multisampling
- 6.7.1 Sample count selection
- 6.7.2 Sample mask
- 6.7.3 Alpha to coverage
- 6.7.4 MSAA resolve operations
- 6.7.5 Custom sample positions (where supported)

## Chapter 7: Render Passes

### 7.1 Render Pass Fundamentals
- 7.1.1 Render pass encoder creation
- 7.1.2 Color attachments
- 7.1.3 Depth/stencil attachment
- 7.1.4 Occlusion queries
- 7.1.5 Timestamp queries

### 7.2 Attachment Operations
- 7.2.1 Load operations: Clear, Load, DontCare
- 7.2.2 Store operations: Store, Discard
- 7.2.3 Clear values (color, depth, stencil)
- 7.2.4 Resolve targets for MSAA
- 7.2.5 Attachment compatibility rules

### 7.3 Render Pass Commands
- 7.3.1 Pipeline binding
- 7.3.2 Bind group binding
- 7.3.3 Vertex buffer binding
- 7.3.4 Index buffer binding
- 7.3.5 Viewport setting
- 7.3.6 Scissor setting
- 7.3.7 Blend constant setting
- 7.3.8 Stencil reference setting
- 7.3.9 Push constants (where supported)

### 7.4 Draw Commands
- 7.4.1 draw(vertex_count, instance_count, first_vertex, first_instance)
- 7.4.2 draw_indexed(index_count, instance_count, first_index, base_vertex, first_instance)
- 7.4.3 draw_indirect(buffer, offset)
- 7.4.4 draw_indexed_indirect(buffer, offset)
- 7.4.5 multi_draw_indirect(buffer, offset, count) [where supported]
- 7.4.6 multi_draw_indexed_indirect(buffer, offset, count) [where supported]
- 7.4.7 multi_draw_indirect_count(buffer, offset, count_buffer, count_offset, max_count) [where supported]

### 7.5 Render Bundles
- 7.5.1 Render bundle encoder
- 7.5.2 Recording commands into bundles
- 7.5.3 Bundle execution
- 7.5.4 Bundle compatibility requirements
- 7.5.5 Bundle caching strategies
- 7.5.6 TRINITY's render bundle system

---

# PART V: COMPUTE PIPELINE

## Chapter 8: Compute Fundamentals

### 8.1 Compute Pipeline
- 8.1.1 Compute pipeline descriptor
- 8.1.2 Pipeline layout association
- 8.1.3 Entry point specification
- 8.1.4 Compute pipeline caching

### 8.2 Compute Shaders
- 8.2.1 @compute entry points
- 8.2.2 @workgroup_size attribute
- 8.2.3 Built-in variables: global_invocation_id, local_invocation_id, workgroup_id, num_workgroups, local_invocation_index
- 8.2.4 Workgroup memory (var<workgroup>)
- 8.2.5 Workgroup synchronization: workgroupBarrier(), storageBarrier()

### 8.3 Compute Pass
- 8.3.1 Compute pass encoder creation
- 8.3.2 Pipeline binding
- 8.3.3 Bind group binding
- 8.3.4 Push constants (where supported)

### 8.4 Dispatch Commands
- 8.4.1 dispatch_workgroups(x, y, z)
- 8.4.2 dispatch_workgroups_indirect(buffer, offset)
- 8.4.3 Workgroup count limits
- 8.4.4 Dispatch sizing strategies

### 8.5 Compute Patterns
- 8.5.1 Parallel reduction
- 8.5.2 Prefix scan (parallel prefix sum)
- 8.5.3 Stream compaction
- 8.5.4 Radix sort
- 8.5.5 Histogram computation
- 8.5.6 Image processing kernels
- 8.5.7 Physics simulation patterns
- 8.5.8 TRINITY's compute library

---

# PART VI: SYNCHRONIZATION & COMMANDS

## Chapter 9: Command Encoding

### 9.1 Command Encoder
- 9.1.1 Command encoder creation
- 9.1.2 Encoder scope and lifetime
- 9.1.3 Pass encoder creation (render, compute)
- 9.1.4 Command buffer finalization

### 9.2 Copy Commands
- 9.2.1 copy_buffer_to_buffer
- 9.2.2 copy_buffer_to_texture
- 9.2.3 copy_texture_to_buffer
- 9.2.4 copy_texture_to_texture
- 9.2.5 Copy alignment requirements
- 9.2.6 Copy region specification

### 9.3 Clear Commands
- 9.3.1 clear_buffer
- 9.3.2 clear_texture (via render pass)
- 9.3.3 Fill patterns

### 9.4 Query Commands
- 9.4.1 Timestamp queries
- 9.4.2 Occlusion queries
- 9.4.3 Pipeline statistics queries (where supported)
- 9.4.4 Query set creation
- 9.4.5 resolve_query_set
- 9.4.6 Query result readback

### 9.5 Debug Commands
- 9.5.1 push_debug_group / pop_debug_group
- 9.5.2 insert_debug_marker
- 9.5.3 Debug labels on resources
- 9.5.4 Integration with GPU debuggers (RenderDoc, PIX, Xcode GPU debugger)

## Chapter 10: Synchronization

### 10.1 Implicit Synchronization
- 10.1.1 wgpu's automatic barrier insertion
- 10.1.2 Resource usage tracking
- 10.1.3 Pass ordering semantics
- 10.1.4 When implicit sync is sufficient

### 10.2 Explicit Synchronization
- 10.2.1 Memory barriers in compute shaders
- 10.2.2 workgroupBarrier()
- 10.2.3 storageBarrier()
- 10.2.4 textureBarrier() (where supported)
- 10.2.5 Full memory barrier

### 10.3 CPU-GPU Synchronization
- 10.3.1 Buffer mapping and callbacks
- 10.3.2 device.poll() semantics
- 10.3.3 Async mapping with futures
- 10.3.4 Fence-like patterns
- 10.3.5 Frame pacing strategies
- 10.3.6 TRINITY's frame synchronization model

### 10.4 Resource State Tracking
- 10.4.1 Resource states in wgpu
- 10.4.2 Transition barriers
- 10.4.3 Split barriers (where supported)
- 10.4.4 TRINITY's frame graph barrier resolution

---

# PART VII: RAY TRACING

## Chapter 11: Acceleration Structures

### 11.1 Acceleration Structure Fundamentals
- 11.1.1 BVH concepts: BLAS and TLAS
- 11.1.2 Bottom-Level Acceleration Structure (BLAS)
- 11.1.3 Top-Level Acceleration Structure (TLAS)
- 11.1.4 Two-level hierarchy rationale
- 11.1.5 wgpu acceleration_structure feature

### 11.2 BLAS Construction
- 11.2.1 Geometry descriptors: triangles, AABBs
- 11.2.2 Triangle geometry: vertex buffer, index buffer, transform
- 11.2.3 AABB geometry: bounding box buffer
- 11.2.4 Geometry flags: OPAQUE, NO_DUPLICATE_ANYHIT_INVOCATION
- 11.2.5 Build flags: PREFER_FAST_TRACE, PREFER_FAST_BUILD, ALLOW_UPDATE, ALLOW_COMPACTION, LOW_MEMORY
- 11.2.6 Scratch buffer allocation
- 11.2.7 Build operation
- 11.2.8 TRINITY's BLAS builder

### 11.3 BLAS Compaction
- 11.3.1 Compaction rationale (memory savings)
- 11.3.2 Compacted size query
- 11.3.3 Compaction copy
- 11.3.4 When to compact vs skip
- 11.3.5 TRINITY's compaction strategy

### 11.4 BLAS Update (Refit)
- 11.4.1 Update vs rebuild trade-offs
- 11.4.2 ALLOW_UPDATE flag requirement
- 11.4.3 Refit operation
- 11.4.4 When refit quality degrades
- 11.4.5 TRINITY's dynamic geometry policy

### 11.5 TLAS Construction
- 11.5.1 Instance descriptor: BLAS reference, transform, instance ID, mask, SBT offset, flags
- 11.5.2 Instance buffer layout
- 11.5.3 Build flags for TLAS
- 11.5.4 Per-frame rebuild strategy
- 11.5.5 Instance culling integration
- 11.5.6 TRINITY's TLAS builder

### 11.6 Acceleration Structure Memory
- 11.6.1 Memory requirements query
- 11.6.2 Scratch memory management
- 11.6.3 Acceleration structure buffer allocation
- 11.6.4 Memory budget tracking
- 11.6.5 TRINITY's AS memory manager

## Chapter 12: Ray Queries (Inline Ray Tracing)

### 12.1 Ray Query Fundamentals
- 12.1.1 ray_query feature
- 12.1.2 Ray query vs ray tracing pipeline
- 12.1.3 Use cases: shadows, AO, simple reflections
- 12.1.4 Performance characteristics

### 12.2 WGSL Ray Query API
- 12.2.1 RayQuery type
- 12.2.2 rayQueryInitialize()
- 12.2.3 rayQueryProceed()
- 12.2.4 rayQueryGetIntersectionType()
- 12.2.5 rayQueryGetCommittedIntersection*() functions
- 12.2.6 rayQueryGetCandidateIntersection*() functions
- 12.2.7 rayQueryConfirmIntersection()
- 12.2.8 rayQueryTerminate()

### 12.3 Ray Flags
- 12.3.1 RAY_FLAG_NONE
- 12.3.2 RAY_FLAG_FORCE_OPAQUE
- 12.3.3 RAY_FLAG_FORCE_NON_OPAQUE
- 12.3.4 RAY_FLAG_TERMINATE_ON_FIRST_HIT
- 12.3.5 RAY_FLAG_SKIP_CLOSEST_HIT_SHADER
- 12.3.6 RAY_FLAG_CULL_BACK_FACING_TRIANGLES
- 12.3.7 RAY_FLAG_CULL_FRONT_FACING_TRIANGLES
- 12.3.8 RAY_FLAG_CULL_OPAQUE
- 12.3.9 RAY_FLAG_CULL_NON_OPAQUE
- 12.3.10 RAY_FLAG_SKIP_TRIANGLES
- 12.3.11 RAY_FLAG_SKIP_AABBS

### 12.4 Ray Query Patterns
- 12.4.1 Shadow ray pattern (any-hit early termination)
- 12.4.2 Closest hit pattern
- 12.4.3 Any-hit with alpha testing (inline)
- 12.4.4 Multi-hit collection (limited)
- 12.4.5 TRINITY's ray query library

## Chapter 13: Ray Tracing Pipelines

### 13.1 RT Pipeline Fundamentals
- 13.1.1 ray_tracing_pipeline feature (experimental)
- 13.1.2 Pipeline vs inline ray queries
- 13.1.3 Shader stages: ray generation, intersection, any-hit, closest-hit, miss
- 13.1.4 Recursion depth

### 13.2 Shader Stages
- 13.2.1 Ray generation shaders (@raygeneration)
- 13.2.2 Intersection shaders (@intersection) — for procedural geometry
- 13.2.3 Any-hit shaders (@anyhit)
- 13.2.4 Closest-hit shaders (@closesthit)
- 13.2.5 Miss shaders (@miss)
- 13.2.6 Callable shaders (@callable)

### 13.3 Hit Groups
- 13.3.1 Hit group concept
- 13.3.2 Triangle hit groups (closest-hit + optional any-hit)
- 13.3.3 Procedural hit groups (intersection + closest-hit + optional any-hit)
- 13.3.4 Hit group indexing

### 13.4 Shader Binding Table (SBT)
- 13.4.1 SBT concept and layout
- 13.4.2 Ray generation record
- 13.4.3 Miss shader records
- 13.4.4 Hit group records
- 13.4.5 Callable shader records
- 13.4.6 SBT indexing calculation
- 13.4.7 SBT stride and alignment
- 13.4.8 TRINITY's SBT builder

### 13.5 RT Pipeline Creation
- 13.5.1 Shader module compilation for RT
- 13.5.2 Pipeline descriptor
- 13.5.3 Max recursion depth
- 13.5.4 Max payload size
- 13.5.5 Max attribute size
- 13.5.6 Pipeline library (where supported)

### 13.6 Ray Tracing Dispatch
- 13.6.1 TraceRay intrinsic
- 13.6.2 Dispatch dimensions
- 13.6.3 Payload passing
- 13.6.4 Attribute passing
- 13.6.5 Recursive tracing

### 13.7 RT Pipeline Patterns
- 13.7.1 Primary ray casting
- 13.7.2 Shadow rays with SBT
- 13.7.3 Reflection rays
- 13.7.4 Refraction rays
- 13.7.5 Ambient occlusion
- 13.7.6 Global illumination (single bounce)
- 13.7.7 Path tracing (multi-bounce)
- 13.7.8 TRINITY's RT effect library

## Chapter 14: RT Advanced Features

### 14.1 Opacity Micromaps (OMM)
- 14.1.1 OMM concept
- 14.1.2 Alpha testing acceleration
- 14.1.3 OMM building
- 14.1.4 OMM integration with BLAS
- 14.1.5 wgpu OMM status

### 14.2 Displacement Micromaps (DMM)
- 14.2.1 DMM concept
- 14.2.2 Micro-geometry detail
- 14.2.3 DMM building
- 14.2.4 DMM integration with BLAS
- 14.2.5 wgpu DMM status

### 14.3 Shader Execution Reordering (SER)
- 14.3.1 SER concept (NVIDIA specific)
- 14.3.2 Coherent ray sorting
- 14.3.3 Performance implications
- 14.3.4 wgpu SER status

### 14.4 Motion Blur
- 14.4.1 Motion BLAS
- 14.4.2 Motion TLAS
- 14.4.3 Time parameter in tracing
- 14.4.4 wgpu motion blur status

---

# PART VIII: ADVANCED RENDERING

## Chapter 15: Indirect Rendering

### 15.1 Indirect Draw
- 15.1.1 DrawIndirect buffer layout
- 15.1.2 DrawIndexedIndirect buffer layout
- 15.1.3 GPU-driven draw call generation
- 15.1.4 Indirect count (where supported)

### 15.2 GPU Culling
- 15.2.1 Frustum culling in compute
- 15.2.2 Occlusion culling with hierarchical-Z
- 15.2.3 GPU-driven LOD selection
- 15.2.4 Indirect buffer compaction
- 15.2.5 TRINITY's GPU culling pipeline

### 15.3 Multi-Draw Indirect
- 15.3.1 multi_draw_indirect feature
- 15.3.2 multi_draw_indexed_indirect feature
- 15.3.3 Batching multiple draws into single call
- 15.3.4 Performance implications

## Chapter 16: Mesh Shaders (Future)

### 16.1 Mesh Shader Fundamentals
- 16.1.1 mesh_shaders feature (not yet in wgpu)
- 16.1.2 Task shader stage
- 16.1.3 Mesh shader stage
- 16.1.4 Meshlet concept

### 16.2 Meshlet Pipeline
- 16.2.1 Meshlet generation
- 16.2.2 Meshlet culling (task shader)
- 16.2.3 Meshlet rendering (mesh shader)
- 16.2.4 Vertex deduplication

### 16.3 TRINITY's Mesh Shader Readiness
- 16.3.1 Meshlet preprocessing
- 16.3.2 Fallback to traditional pipeline
- 16.3.3 Abstraction layer

## Chapter 17: Bindless Resources

### 17.1 Bindless Fundamentals
- 17.1.1 Bindless texture arrays
- 17.1.2 Bindless buffer arrays
- 17.1.3 Descriptor indexing
- 17.1.4 Non-uniform indexing

### 17.2 Implementation Patterns
- 17.2.1 Texture atlas approach
- 17.2.2 Texture array approach
- 17.2.3 Storage buffer indirection
- 17.2.4 Hybrid approaches

### 17.3 TRINITY's Bindless System
- 17.3.1 Texture registry
- 17.3.2 Buffer registry
- 17.3.3 Material table
- 17.3.4 Index allocation and recycling

---

# PART IX: PRESENTATION

## Chapter 18: Surface & Swapchain

### 18.1 Surface
- 18.1.1 Surface creation from window handle
- 18.1.2 Surface capabilities query
- 18.1.3 Supported formats
- 18.1.4 Supported present modes
- 18.1.5 Supported alpha modes

### 18.2 Surface Configuration
- 18.2.1 Format selection
- 18.2.2 Present mode: Immediate, Mailbox, Fifo, FifoRelaxed
- 18.2.3 Alpha mode: Auto, Opaque, PreMultiplied, PostMultiplied
- 18.2.4 Width and height
- 18.2.5 View formats for sRGB reinterpretation

### 18.3 Frame Acquisition
- 18.3.1 get_current_texture()
- 18.3.2 SurfaceTexture handling
- 18.3.3 Suboptimal and outdated surfaces
- 18.3.4 Surface reconfiguration on resize

### 18.4 Presentation
- 18.4.1 present() call
- 18.4.2 Vsync and frame pacing
- 18.4.3 Triple buffering strategies
- 18.4.4 TRINITY's presentation engine

---

# PART X: PLATFORM & BACKENDS

## Chapter 19: Platform Considerations

### 19.1 Vulkan Backend
- 19.1.1 Vulkan instance/device mapping
- 19.1.2 Vulkan extension requirements
- 19.1.3 Vulkan-specific features
- 19.1.4 Debugging with validation layers

### 19.2 Metal Backend
- 19.2.1 Metal device selection
- 19.2.2 Metal feature sets
- 19.2.3 Metal-specific considerations
- 19.2.4 Argument buffers for bindless

### 19.3 DX12 Backend
- 19.3.1 DX12 device selection
- 19.3.2 DX12 feature levels
- 19.3.3 Root signature mapping
- 19.3.4 DX12-specific features

### 19.4 WebGPU Backend
- 19.4.1 Browser compatibility
- 19.4.2 WebGPU spec conformance
- 19.4.3 Web-specific limitations
- 19.4.4 WASM integration

### 19.5 OpenGL Backend (Fallback)
- 19.5.1 OpenGL ES / WebGL fallback
- 19.5.2 Feature limitations
- 19.5.3 Performance considerations

## Chapter 20: Feature Detection & Capability Abstraction

### 20.1 Feature Flags
- 20.1.1 Core features (always available)
- 20.1.2 Optional features (adapter-dependent)
- 20.1.3 Experimental features (unstable API)
- 20.1.4 Feature dependency chains

### 20.2 Limits
- 20.2.1 max_texture_dimension_*
- 20.2.2 max_buffer_size
- 20.2.3 max_bind_groups
- 20.2.4 max_bindings_per_bind_group
- 20.2.5 max_storage_buffers_per_shader_stage
- 20.2.6 max_compute_workgroups_per_dimension
- 20.2.7 max_compute_workgroup_size_*
- 20.2.8 And many more...

### 20.3 TRINITY's Capability System
- 20.3.1 Capability enum (MINIMAL, STANDARD, ADVANCED, FULL)
- 20.3.2 Feature requirement specification per render path
- 20.3.3 Automatic fallback selection
- 20.3.4 Runtime capability queries

---

# PART XI: DEBUGGING & PROFILING

## Chapter 21: Debugging

### 21.1 wgpu Debugging Features
- 21.1.1 Validation (WGPU_VALIDATION)
- 21.1.2 Debug markers and groups
- 21.1.3 Object labels
- 21.1.4 Error scopes

### 21.2 External Debuggers
- 21.2.1 RenderDoc integration
- 21.2.2 PIX for Windows
- 21.2.3 Xcode GPU Frame Capture
- 21.2.4 NVIDIA Nsight Graphics
- 21.2.5 AMD Radeon GPU Profiler

### 21.3 TRINITY's Debug System
- 21.3.1 Debug visualization modes
- 21.3.2 Resource inspection
- 21.3.3 Pipeline state dump
- 21.3.4 Frame capture triggers

## Chapter 22: Profiling

### 22.1 GPU Timing
- 22.1.1 Timestamp queries
- 22.1.2 Pass timing
- 22.1.3 Compute kernel timing
- 22.1.4 Timer resolution

### 22.2 Pipeline Statistics
- 22.2.1 Vertex shader invocations
- 22.2.2 Fragment shader invocations
- 22.2.3 Compute shader invocations
- 22.2.4 Clipping statistics

### 22.3 Memory Profiling
- 22.3.1 Resource memory tracking
- 22.3.2 Memory budget monitoring
- 22.3.3 Memory leak detection
- 22.3.4 Memory pressure handling

### 22.4 TRINITY's Profiling System
- 22.4.1 Per-pass timing
- 22.4.2 Memory dashboard
- 22.4.3 Draw call statistics
- 22.4.4 Bottleneck analysis

---

# PART XII: TRINITY INTEGRATION

## Chapter 23: Frame Graph Integration

### 23.1 Resource Declaration
- 23.1.1 Virtual resources
- 23.1.2 Transient resources
- 23.1.3 External resources
- 23.1.4 Resource aliasing

### 23.2 Pass Declaration
- 23.2.1 Render passes
- 23.2.2 Compute passes
- 23.2.3 Ray tracing passes
- 23.2.4 Copy passes

### 23.3 Barrier Resolution
- 23.3.1 Automatic barrier placement
- 23.3.2 Resource state tracking
- 23.3.3 Barrier batching
- 23.3.4 Aliasing barriers

### 23.4 Execution
- 23.4.1 Pass scheduling
- 23.4.2 Async compute overlap
- 23.4.3 Resource lifetime management
- 23.4.4 Frame-to-frame resource recycling

## Chapter 24: Python Bridge

### 24.1 PyO3 Binding Layer
- 24.1.1 Type marshalling
- 24.1.2 Handle management
- 24.1.3 Callback patterns
- 24.1.4 Error propagation

### 24.2 Resource Descriptors
- 24.2.1 Python-side descriptors
- 24.2.2 Descriptor validation
- 24.2.3 Descriptor to wgpu translation
- 24.2.4 Descriptor caching

### 24.3 Command Recording
- 24.3.1 Python command builder
- 24.3.2 Deferred execution
- 24.3.3 Command batching
- 24.3.4 Error handling

---

# APPENDICES

## Appendix A: wgpu Feature Matrix

| Feature | Vulkan | Metal | DX12 | WebGPU | Status |
|---------|--------|-------|------|--------|--------|
| Core | ✓ | ✓ | ✓ | ✓ | Stable |
| Compute | ✓ | ✓ | ✓ | ✓ | Stable |
| Ray Query | ✓ | ✗ | ✗ | ✗ | Stable |
| RT Pipeline | ✓ | ✗ | ✗ | ✗ | Experimental |
| Mesh Shaders | ✗ | ✗ | ✗ | ✗ | Not yet |
| ... | | | | | |

## Appendix B: WGSL Quick Reference

(Condensed syntax reference for common patterns)

## Appendix C: Glossary

- **BLAS**: Bottom-Level Acceleration Structure
- **TLAS**: Top-Level Acceleration Structure
- **SBT**: Shader Binding Table
- **PSO**: Pipeline State Object
- **BVH**: Bounding Volume Hierarchy
- **MSAA**: Multisample Anti-Aliasing
- **HiZ**: Hierarchical Z-buffer
- **LOD**: Level of Detail

## Appendix D: Version History

| Version | wgpu | Key Changes |
|---------|------|-------------|
| 25.0 | | Ray query stable |
| 26.0 | | RT pipeline (expected) |
| 27.0 | | Mesh shaders (speculative) |

---

# Cross-References to GAPSET Documents

This TOC covers the COMPLETE wgpu surface. The existing GAPSET documents map to specific chapters:

| GAPSET | TOC Coverage |
|--------|-------------|
| GAPSET_1 (Frame Graph) | Part XII, Chapter 23 |
| GAPSET_6 (GI/Reflections) | Part VII (RT), Part VIII (Compute patterns) |
| GAPSET_9 (Ray Tracing) | Part VII (Chapters 11-14) |
| GAPSET_12 (RHI) | Part I-VI (full device/resource/pipeline stack) |

---

*End of WGPU_TOC.md*
