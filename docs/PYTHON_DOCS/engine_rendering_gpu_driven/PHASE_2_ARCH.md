# PHASE 2 ARCHITECTURE: Meshlet/Cluster System

## Overview

Phase 2 covers the meshlet/cluster rendering system implemented in `meshlet.py` (731 lines). Meshlets partition meshes into GPU-friendly clusters that enable per-cluster culling and mesh shader amplification.

## Components

### Core Data Structures

| Type | Purpose |
|------|---------|
| `MeshletBounds` | Bounding sphere + normal cone for culling |
| `Meshlet` | Vertex indices, local indices, triangle list |
| `MeshletLODChain` | Hierarchical LOD levels of meshlets |
| `MeshletMesh` | Complete mesh with meshlet partitioning |

### Processing Pipeline

```
Input: Vertex buffer + Index buffer
    |
    v
[MeshletBuilder] -- Greedy clustering
    |               64 vertex limit, 124 triangle limit
    |               Maximize vertex reuse
    v
[BoundsComputation] -- Ritter refinement for bounding sphere
    |                  Normal cone from face normals
    v
[LODGeneration] -- Simplify mesh
    |              Build MeshletLODChain
    v
[MeshletCuller] -- Backface cone test
    |              Per-cluster visibility
    v
Output: Visible meshlet indices
```

## Architecture Decisions

### ADR-MESH-001: 64/124 Meshlet Limits

**Context**: Meshlets need size limits for GPU efficiency.

**Decision**: Use 64 vertices / 124 triangles per meshlet (industry standard).

**Rationale**:
- Matches Vulkan mesh shader primitive limits
- Fits in GPU L1 cache (64 vertices x 48 bytes = 3KB)
- 124 triangles allow 372 indices (fits in single cache line batch)
- Compatible with Nanite, Unreal, AMD GPUOpen meshlet tools

**Implementation**:
```python
MESHLET_MAX_VERTICES = 64
MESHLET_MAX_TRIANGLES = 124
MESHLET_MAX_INDICES = 378  # 124 * 3 + 6 padding
```

### ADR-MESH-002: Greedy Clustering Algorithm

**Context**: Need algorithm to partition mesh into meshlets.

**Decision**: Use greedy clustering with vertex reuse optimization.

**Rationale**:
- O(n) complexity (linear in triangle count)
- Prioritizes vertices already in current meshlet
- Adjacency-aware: prefers neighboring triangles
- Good balance between partition quality and build speed

**Implementation** (lines 280-380):
```python
def _build_meshlet(self, start_triangle: int) -> Meshlet:
    while meshlet.triangle_count < MESHLET_MAX_TRIANGLES:
        best_triangle = self._find_best_triangle(meshlet)
        if not self._can_add_triangle(meshlet, best_triangle):
            break
        self._add_triangle(meshlet, best_triangle)
```

### ADR-MESH-003: Ritter Refinement for Bounding Spheres

**Context**: Need tight bounding spheres for meshlet culling.

**Decision**: Use Ritter's algorithm with refinement passes.

**Rationale**:
- Simple O(n) initial sphere (diameter between extreme points)
- Refinement passes tighten sphere (iterate until convergence)
- Typically 2-3 passes for near-optimal sphere
- Avoids expensive Welzl's algorithm (recursive, O(n) expected but O(n^2) worst)

**Implementation**:
```python
def _compute_bounding_sphere(self, meshlet: Meshlet) -> (Vec3, float):
    # Initial sphere from diameter
    center, radius = self._initial_sphere(points)
    # Ritter refinement
    for _ in range(MAX_REFINEMENT_PASSES):
        center, radius = self._refine_sphere(center, radius, points)
    return center, radius
```

### ADR-MESH-004: Normal Cone Backface Culling

**Context**: Need efficient per-cluster backface culling.

**Decision**: Compute normal cone (axis + cutoff) for each meshlet.

**Rationale**:
- Single dot product test per cluster (vs per-triangle)
- `cone_axis`: average normal direction
- `cone_cutoff`: cosine of maximum deviation from axis
- If `dot(view_dir, axis) < -cutoff`, all faces are backfacing

**Implementation** (lines 416-467):
```python
def _compute_normal_cone(self, meshlet: Meshlet) -> (Vec3, float):
    # Average all face normals
    avg_normal = sum(face_normals) / len(face_normals)
    # Find minimum dot product
    min_dot = min(dot(n, avg_normal) for n in face_normals)
    return avg_normal, min_dot
```

### ADR-MESH-005: Hierarchical LOD Chain

**Context**: Meshlets need LOD support for distance-based detail.

**Decision**: Build MeshletLODChain with progressively simplified meshlet sets.

**Rationale**:
- Each LOD level is independent meshlet set (no morphing)
- LOD selection at meshlet granularity (not whole mesh)
- Supports mixed-LOD rendering (different clusters at different LODs)
- Compatible with virtual geometry streaming

**Implementation**:
```python
class MeshletLODChain:
    levels: list[list[Meshlet]]  # LOD 0 = highest detail
    lod_distances: list[float]   # Transition distances
```

## Data Flow

### Input Data

```python
class MeshData:
    positions: list[Vec3]
    normals: list[Vec3]
    uvs: list[Vec2]
    indices: list[int]
```

### Meshlet Data

```python
class Meshlet:
    vertex_offset: int      # Offset into global vertex buffer
    vertex_count: int       # Number of vertices in meshlet
    triangle_offset: int    # Offset into local index buffer
    triangle_count: int     # Number of triangles
    local_indices: list[int]  # Indices into meshlet's vertex list
    
class MeshletBounds:
    center: Vec3
    radius: float
    cone_axis: Vec3
    cone_cutoff: float
```

### Output Data

```python
class CullResult:
    visible_meshlets: list[int]  # Indices of visible meshlets
    total_triangles: int         # Sum of triangle counts
```

## GPU Port Considerations

### Mesh Shader Mapping

| Stage | Shader Type | Invocations |
|-------|-------------|-------------|
| MeshletCuller | Task Shader | 1 thread per meshlet |
| VertexTransform | Mesh Shader | 1 thread per vertex |
| PrimitiveOutput | Mesh Shader | 1 thread per triangle |

### Buffer Layout

```wgsl
struct Meshlet {
    vertex_offset: u32,
    vertex_count: u32,
    triangle_offset: u32,
    triangle_count: u32,
}

struct MeshletBounds {
    center: vec3<f32>,
    radius: f32,
    cone_axis: vec3<f32>,
    cone_cutoff: f32,
}

struct MeshletPayload {
    meshlet_indices: array<u32, MAX_MESHLETS_PER_GROUP>,
    meshlet_count: u32,
}
```

### Memory Layout

For optimal GPU access:
1. **Meshlet descriptors** - Structure of Arrays for coalesced reads
2. **Vertex data** - Interleaved or separate streams (benchmark)
3. **Local indices** - Packed u8 or u16 (124 triangles x 3 = 372 indices)

## Dependencies

- Requires mesh import from asset pipeline
- Provides meshlet data for visibility buffer pass
- Integrates with culling pipeline for per-cluster culling
