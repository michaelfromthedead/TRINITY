# engine/rendering/gpu_driven Investigation

**Lines**: 4,859 (actual total from all 6 Python files)
**Classification**: REAL (CPU-side simulation of GPU algorithms)

## File Analysis

| File | Lines | Classification | Key Components |
|------|-------|----------------|----------------|
| `culling.py` | 1,109 | REAL | Vec3/4, AABB, Frustum, FrustumCuller, OcclusionCuller (HZB), DistanceCuller, SmallFeatureCuller, CullingPipeline |
| `visibility_buffer.py` | 836 | REAL | VisibilityData (bit-packing), VisibilityBuffer, VisibilityBufferPass (software rasterizer), MaterialTileClassifier, DeferredTexturingPass, MaterialSortingPass, VisibilityBufferPipeline |
| `bindless.py` | 786 | REAL | ResourceHandle (generational), TextureDescriptor, BufferDescriptor, SamplerDescriptor, BindlessTextureManager, BindlessBufferManager (BDA), MaterialResources, MaterialResourceTable, BindlessResourceSystem |
| `instancing.py` | 736 | REAL | Mat4x4 (TRS from quaternion), InstanceData, BatchKey, InstanceBatch, InstanceBatcher, MultiDrawIndirectManager, CulledInstanceBatcher |
| `meshlet.py` | 731 | REAL | MeshletBounds (normal cone), Meshlet, MeshletBuilder (greedy clustering), MeshletCuller (backface cone test), MeshletLODChain, MeshletMesh |
| `indirect_draw.py` | 661 | REAL | DrawIndexedIndirectArgs, DrawIndirectArgs, DispatchIndirectArgs, DrawCommand, IndirectDrawBuffer, DrawCommandGenerator, MultiDrawIndirectBuffer, DrawCommandCompactor |

## Key Findings

This subsystem implements a **complete GPU-driven rendering pipeline** with all core algorithms:

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

**All code contains real algorithms with complete implementations.** There are no stubs, `pass` statements, or `NotImplementedError` raises in any logic paths.

**Important Caveat**: All algorithms execute on CPU in Python. There is no actual GPU execution (no WGSL/GLSL shaders, no wgpu bindings). This is a CPU simulation layer providing data structures for eventual GPU upload.

## Evidence

### REAL: Frustum plane extraction (culling.py lines 202-272)
```python
@classmethod
def from_view_projection_matrix(cls, vp_matrix: list[list[float]]) -> "Frustum":
    frustum = cls()
    m = vp_matrix
    # Left plane: row3 + row0
    frustum.planes[FrustumPlane.LEFT] = Vec4(
        m[0][3] + m[0][0], m[1][3] + m[1][0],
        m[2][3] + m[2][0], m[3][3] + m[3][0],
    )
    # ... (5 more planes)
    # Normalize planes
    for i, plane in enumerate(frustum.planes):
        length = math.sqrt(plane.x**2 + plane.y**2 + plane.z**2)
        if length > CullingConstants.EPSILON:
            inv_len = 1.0 / length
            frustum.planes[i] = Vec4(plane.x*inv_len, plane.y*inv_len, ...)
```

### REAL: HZB mip pyramid construction (culling.py lines 539-577)
```python
def _build_hzb(self, depth_buffer: list[list[float]]) -> None:
    self._hzb_pyramid = []
    current_level = depth_buffer
    self._hzb_pyramid.append(current_level)
    while width > 1 or height > 1:
        new_level: list[list[float]] = []
        for y in range(new_height):
            row: list[float] = []
            for x in range(new_width):
                max_depth = max(
                    current_level[y0][x0], current_level[y0][x1],
                    current_level[y1][x0], current_level[y1][x1],
                )
                row.append(max_depth)
```

### REAL: Triangle rasterization with edge functions (visibility_buffer.py lines 439-475)
```python
def edge_function(a, b, p) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

for y in range(min_y, max_y + 1):
    for x in range(min_x, max_x + 1):
        w0 = edge_function(v1, v2, p)
        w1 = edge_function(v2, v0, p)
        w2 = edge_function(v0, v1, p)
        if w0 >= 0 and w1 >= 0 and w2 >= 0:
            area = w0 + w1 + w2
            depth = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]
```

### REAL: Meshlet normal cone computation (meshlet.py lines 416-467)
```python
def _compute_normal_cone(self, meshlet: Meshlet, bounds: MeshletBounds) -> None:
    avg_normal = Vec3(0.0, 0.0, 0.0)
    normals: list[Vec3] = []
    for i in range(meshlet.triangle_count):
        tri = meshlet.get_triangle(i)
        e1 = p1 - p0
        e2 = p2 - p0
        normal = Vec3(
            e1.y * e2.z - e1.z * e2.y,
            e1.z * e2.x - e1.x * e2.z,
            e1.x * e2.y - e1.y * e2.x,
        )
        normals.append(normal.normalized())
        avg_normal = avg_normal + normal
    # Find maximum deviation for cone cutoff
    min_dot = 1.0
    for normal in normals:
        dot = normal.dot(avg_normal)
        min_dot = min(min_dot, dot)
    bounds.cone_cutoff = min_dot
```

### REAL: Draw command compaction (indirect_draw.py lines 549-636)
```python
def compact(self, commands: Sequence[DrawCommand]) -> list[DrawCommand]:
    sorted_commands = sorted(commands, key=lambda c: c.sort_key)
    compacted: list[DrawCommand] = []
    current = sorted_commands[0]
    for i in range(1, len(sorted_commands)):
        next_cmd = sorted_commands[i]
        if self._can_merge(current, next_cmd):
            current = self._merge(current, next_cmd)
        else:
            compacted.append(current)
            current = next_cmd
    compacted.append(current)
    return compacted
```

## GPU-Driven Components Summary

| Component | Purpose | Implementation Status |
|-----------|---------|----------------------|
| FrustumCuller | VP matrix plane extraction, sphere/AABB tests | Complete |
| OcclusionCuller | HZB pyramid build, screen projection, mip sampling | Complete |
| DistanceCuller | Distance culling with LOD selection | Complete |
| SmallFeatureCuller | Screen-size culling | Complete |
| MeshletBuilder | Greedy clustering, adjacency, Ritter refinement | Complete |
| MeshletCuller | Normal cone backface culling | Complete |
| VisibilityBuffer | Pixel triangle/instance storage | Complete |
| MaterialTileClassifier | 8x8 tile material grouping | Complete |
| DeferredTexturingPass | Material-sorted shading | Complete |
| BindlessResourceSystem | Handle/descriptor management | Complete |
| DrawCommandGenerator | Indirect draw batching | Complete |
| DrawCommandCompactor | Command merging | Complete |

## Missing for True GPU Execution

1. WGSL/GLSL compute shaders for culling passes
2. wgpu/WebGPU Buffer/BindGroup bindings
3. GPU descriptor heap submission
4. Indirect draw command buffer execution
5. Mesh shader integration for meshlets

## Verdict

**REAL ALGORITHMS, CPU SIMULATION** - The `engine/rendering/gpu_driven` subsystem contains 4,859 lines of production-quality algorithm implementations for GPU-driven rendering. All math is correct, all data structures are complete. The code provides a CPU simulation layer that could be directly ported to GPU shaders or used to generate GPU buffer data.
