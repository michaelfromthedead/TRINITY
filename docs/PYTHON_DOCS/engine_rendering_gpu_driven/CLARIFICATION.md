# CLARIFICATION: engine/rendering/gpu_driven

## Philosophical Framing

### Why GPU-Driven Rendering?

Traditional rendering pipelines suffer from CPU bottlenecks due to per-object draw calls. GPU-driven rendering inverts this model: the GPU performs culling, LOD selection, and draw command generation, while the CPU merely uploads data and issues a single indirect draw call.

This subsystem implements all the foundational algorithms required for GPU-driven rendering, albeit as CPU simulations. The design philosophy is:

1. **Algorithm Correctness First** - Implement mathematically correct algorithms that can be verified and debugged on CPU
2. **GPU-Compatible Data Structures** - Design data structures that map directly to GPU buffer layouts
3. **Component Modularity** - Each component (culling, visibility, bindless) can be ported independently

### The CPU Simulation Approach

The decision to implement GPU algorithms in Python serves multiple purposes:

1. **Prototyping** - Algorithm behavior can be validated before shader implementation
2. **Reference Implementation** - Provides ground truth for GPU shader validation
3. **Educational** - Clear Python code documents the algorithms without shader syntax noise
4. **Test Data Generation** - Can generate expected outputs for GPU shader unit tests

## Design Rationale

### Culling Pipeline Architecture

The culling pipeline is designed as a multi-stage sequence:

```
FrustumCuller -> OcclusionCuller -> DistanceCuller -> SmallFeatureCuller
```

**Why this order?**

1. **Frustum culling** is the cheapest test (plane-sphere/AABB) and eliminates the most objects
2. **Occlusion culling** (HZB) requires screen projection, so frustum-visible objects are tested
3. **Distance culling** integrates LOD selection for surviving objects
4. **Small feature culling** is a final quality pass to eliminate sub-pixel geometry

This order maximizes early rejection while minimizing expensive tests on objects that will be culled anyway.

### Visibility Buffer vs Deferred Rendering

The visibility buffer approach (as pioneered by Nanite) offers advantages over traditional deferred rendering:

1. **Reduced Overdraw** - Only visible triangles are shaded
2. **Material Efficiency** - Material tile classification groups pixels by material for coherent shading
3. **Decoupling** - Geometry pass (visibility) is decoupled from shading pass

The 32-bit visibility data format (12-bit triangle + 20-bit instance) supports:
- Up to 4,096 triangles per draw (sufficient for meshlet-based rendering)
- Up to 1,048,576 instances per frame

### Bindless Resource Philosophy

Traditional rendering binds resources per-draw. Bindless rendering uploads all resources to descriptor heaps and indexes them dynamically:

1. **Reduced State Changes** - No per-draw bind calls
2. **Dynamic Indexing** - Shaders can access any resource via handle
3. **Generational Handles** - Detect stale references to freed resources

The generational handle design (index + generation) prevents use-after-free bugs common in resource management systems.

### Meshlet/Cluster Rendering

Meshlet rendering partitions meshes into small clusters (64 vertices, 124 triangles) that enable:

1. **Per-Cluster Culling** - Backface cull entire clusters via normal cones
2. **LOD Streaming** - Load/unload LOD levels at meshlet granularity
3. **GPU Amplification** - Mesh shaders can expand/reject meshlets

The greedy clustering algorithm maximizes vertex reuse within cache limits, improving GPU vertex cache efficiency.

## Key Design Decisions

### Gribb-Hartmann Plane Extraction

Frustum planes are extracted from the combined view-projection matrix using the Gribb-Hartmann method. This approach:

- Avoids explicit near/far plane calculation
- Works correctly with arbitrary projection matrices
- Produces normalized planes for distance queries

### HZB Mip Pyramid

The Hierarchical Z-Buffer uses max-reduction (not min) because:

- Depth buffer uses reverse-Z (near = 1.0, far = 0.0)
- Max-reduction preserves the closest depth at each mip
- Conservative occlusion test: if max(tile) < object depth, object is occluded

### Edge Function Rasterization

The software rasterizer uses edge functions for inside/outside tests:

```
edge(a, b, p) = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)
```

This is the same algorithm used in GPU hardware rasterizers, ensuring behavioral equivalence.

### Normal Cone Backface Culling

Meshlet normal cones encode the range of face normals within a cluster:

- `cone_axis`: average normal direction
- `cone_cutoff`: minimum dot product with any face normal

If `dot(view_dir, cone_axis) < -cone_cutoff`, the entire meshlet is backfacing and can be culled.

## Integration Points

### With Rendering Backend

The GPU-driven components provide data for the rendering backend:

1. **Culling Pipeline** -> Visibility bitmask for instances
2. **Visibility Buffer** -> Triangle/instance IDs for deferred shading
3. **Bindless System** -> Descriptor indices for material lookup
4. **Indirect Draw** -> GPU command buffers

### With Asset Pipeline

Meshlet building requires mesh preprocessing:

1. **Mesh Import** -> Vertex/index buffers
2. **Meshlet Builder** -> Cluster partitioning
3. **LOD Generation** -> MeshletLODChain construction
4. **Bounds Computation** -> Bounding spheres and normal cones

## Future Considerations

### GPU Shader Port

Each component maps to GPU compute shaders:

| Component | Shader Type | Dispatch |
|-----------|-------------|----------|
| FrustumCuller | Compute | 1 thread per instance |
| OcclusionCuller | Compute | 1 thread per instance |
| MeshletCuller | Compute/Mesh | 1 thread per meshlet |
| VisibilityBufferPass | Fragment | Per-pixel |
| DrawCommandGenerator | Compute | 1 thread per batch |

### Mesh Shader Integration

For full Nanite-style rendering, meshlet culling should use mesh shaders:

1. Task shader: per-meshlet culling, outputs surviving meshlet count
2. Mesh shader: vertex transformation, primitive output

This requires mesh shader support (Vulkan 1.3, DX12 Ultimate, WebGPU extension).
