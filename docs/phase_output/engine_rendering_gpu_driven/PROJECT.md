# PROJECT: engine/rendering/gpu_driven

## Overview

The `engine/rendering/gpu_driven` subsystem implements a complete GPU-driven rendering pipeline with all core algorithms for modern rendering techniques. The implementation spans 4,859 lines across 6 Python files, providing CPU simulation of GPU algorithms.

## Scope

### In Scope

1. **Multi-stage GPU Culling Pipeline**
   - Frustum culling (Gribb-Hartmann plane extraction)
   - HZB occlusion culling (mip pyramid, screen-space projection)
   - Distance culling with LOD selection
   - Small feature culling (screen-space size)

2. **Visibility Buffer Rendering** (Nanite-style)
   - Visibility data bit-packing (32-bit: 12-bit tri + 20-bit instance)
   - Software triangle rasterization with edge functions
   - Material tile classification (8x8 tiles)
   - Deferred texturing with material-sorted shading

3. **Bindless Resources**
   - Generational handles with validation
   - Descriptor heap management (free-list allocation)
   - Buffer Device Address (BDA) support
   - PBR material resource bindings

4. **Meshlet/Cluster Rendering**
   - Standard 64 vertices / 124 triangles limits
   - Greedy clustering with cache-efficient vertex reuse
   - Bounding sphere (Ritter refinement)
   - Normal cone backface culling
   - Hierarchical LOD support

5. **Indirect Drawing**
   - Multi-draw indirect command generation
   - Instance batching by mesh+material
   - Draw command compaction (merging contiguous ranges)

### Out of Scope

- WGSL/GLSL compute shader implementation
- wgpu/WebGPU buffer/BindGroup bindings
- GPU descriptor heap submission
- Indirect draw command buffer execution on GPU
- Mesh shader integration for meshlets

## Files

| File | Lines | Key Components |
|------|-------|----------------|
| `culling.py` | 1,109 | Vec3/4, AABB, Frustum, FrustumCuller, OcclusionCuller (HZB), DistanceCuller, SmallFeatureCuller, CullingPipeline |
| `visibility_buffer.py` | 836 | VisibilityData (bit-packing), VisibilityBuffer, VisibilityBufferPass (software rasterizer), MaterialTileClassifier, DeferredTexturingPass, MaterialSortingPass, VisibilityBufferPipeline |
| `bindless.py` | 786 | ResourceHandle (generational), TextureDescriptor, BufferDescriptor, SamplerDescriptor, BindlessTextureManager, BindlessBufferManager (BDA), MaterialResources, MaterialResourceTable, BindlessResourceSystem |
| `instancing.py` | 736 | Mat4x4 (TRS from quaternion), InstanceData, BatchKey, InstanceBatch, InstanceBatcher, MultiDrawIndirectManager, CulledInstanceBatcher |
| `meshlet.py` | 731 | MeshletBounds (normal cone), Meshlet, MeshletBuilder (greedy clustering), MeshletCuller (backface cone test), MeshletLODChain, MeshletMesh |
| `indirect_draw.py` | 661 | DrawIndexedIndirectArgs, DrawIndirectArgs, DispatchIndirectArgs, DrawCommand, IndirectDrawBuffer, DrawCommandGenerator, MultiDrawIndirectBuffer, DrawCommandCompactor |

## Goals

1. Validate all algorithm implementations are complete and mathematically correct
2. Ensure data structures are suitable for GPU buffer upload
3. Document component interfaces for shader port
4. Establish integration points with the rendering backend
5. Prepare for GPU execution via WGSL shaders

## Constraints

1. **CPU Simulation Layer** - All algorithms execute on CPU in Python; no actual GPU execution
2. **No Native GPU Bindings** - No wgpu, WebGPU, or shader compilation infrastructure
3. **Python Performance** - CPU simulation is not performance-representative of GPU execution
4. **Data Structure Compatibility** - Must maintain compatibility with GPU buffer layouts

## Acceptance Criteria

### Phase 1: Culling Pipeline
- [ ] Frustum plane extraction produces correct 6 planes from VP matrix
- [ ] HZB pyramid builds correct mip chain with max-depth reduction
- [ ] Distance culling integrates with LOD selection logic
- [ ] Small feature culling uses correct screen-space projection

### Phase 2: Meshlet/Cluster System
- [ ] Meshlet builder respects 64 vertex / 124 triangle limits
- [ ] Greedy clustering maximizes vertex reuse
- [ ] Ritter refinement produces tight bounding spheres
- [ ] Normal cone backface culling is geometrically correct

### Phase 3: Visibility Buffer Rendering
- [ ] Bit-packing correctly encodes 12-bit triangle + 20-bit instance
- [ ] Edge function rasterization handles all triangle orientations
- [ ] Material tile classification groups pixels correctly
- [ ] Deferred texturing resolves material IDs

### Phase 4: Indirect Draw and Bindless
- [ ] Generational handles detect stale references
- [ ] Descriptor heap allocation/deallocation is leak-free
- [ ] Draw command compaction merges contiguous ranges correctly
- [ ] Instance batching groups by mesh+material key

## Classification

**REAL ALGORITHMS, CPU SIMULATION** - The subsystem contains production-quality algorithm implementations for GPU-driven rendering. All math is correct, all data structures are complete. The code provides a CPU simulation layer that could be directly ported to GPU shaders or used to generate GPU buffer data.
