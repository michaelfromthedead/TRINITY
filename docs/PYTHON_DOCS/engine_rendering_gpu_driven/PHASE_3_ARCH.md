# PHASE 3 ARCHITECTURE: Visibility Buffer Rendering

## Overview

Phase 3 covers the visibility buffer rendering system implemented in `visibility_buffer.py` (836 lines). This implements Nanite-style rendering where geometry is decoupled from shading via a visibility buffer.

## Components

### Core Data Structures

| Type | Purpose |
|------|---------|
| `VisibilityData` | 32-bit packed triangle/instance ID |
| `VisibilityBuffer` | 2D buffer of VisibilityData per pixel |
| `MaterialTile` | 8x8 tile with material classification |
| `ShadingResult` | Final color output per pixel |

### Rendering Pipeline

```
Input: Meshlet stream + Instance transforms
    |
    v
[VisibilityBufferPass] -- Software triangle rasterization
    |                     Edge function coverage test
    |                     Depth test + atomic visibility write
    v
[MaterialTileClassifier] -- 8x8 tile classification
    |                       Group pixels by material ID
    v
[MaterialSortingPass] -- Sort tiles by material
    |                    Generate deferred shading jobs
    v
[DeferredTexturingPass] -- Material-sorted shading
    |                      Reconstruct position from depth
    |                      Texture fetch + lighting
    v
Output: Final color buffer
```

## Architecture Decisions

### ADR-VISBUF-001: 32-bit Visibility Data Format

**Context**: Need to store triangle and instance ID per pixel efficiently.

**Decision**: Use 32-bit format: 12-bit triangle ID + 20-bit instance ID.

**Rationale**:
- 12 bits = 4,096 triangles per draw (sufficient for 124-tri meshlets)
- 20 bits = 1,048,576 instances per frame (sufficient for large scenes)
- Single atomic write per pixel (no read-modify-write)
- Compatible with GPU atomic operations

**Implementation**:
```python
class VisibilityData:
    def pack(self, triangle_id: int, instance_id: int) -> int:
        return (instance_id << 12) | triangle_id
    
    def unpack(self, packed: int) -> tuple[int, int]:
        triangle_id = packed & 0xFFF
        instance_id = packed >> 12
        return triangle_id, instance_id
```

### ADR-VISBUF-002: Edge Function Rasterization

**Context**: Need software rasterization for visibility buffer fill.

**Decision**: Use edge function (signed area) for inside/outside test.

**Rationale**:
- Same algorithm as GPU hardware rasterizers
- Parallel evaluation per pixel (no dependencies)
- Handles all triangle orientations correctly
- Supports barycentric interpolation

**Implementation** (lines 439-475):
```python
def edge_function(a, b, p) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

for y in range(min_y, max_y + 1):
    for x in range(min_x, max_x + 1):
        w0 = edge_function(v1, v2, p)
        w1 = edge_function(v2, v0, p)
        w2 = edge_function(v0, v1, p)
        if w0 >= 0 and w1 >= 0 and w2 >= 0:
            # Inside triangle
```

### ADR-VISBUF-003: 8x8 Material Tile Classification

**Context**: Deferred shading needs material coherence for efficiency.

**Decision**: Classify pixels into 8x8 tiles by material ID.

**Rationale**:
- 8x8 matches GPU wavefront size (64 threads)
- Single material per tile enables uniform branching
- Mixed-material tiles handled via multi-pass
- Reduces texture cache thrashing

**Implementation**:
```python
class MaterialTileClassifier:
    TILE_SIZE = 8
    
    def classify_tile(self, tile_x: int, tile_y: int) -> set[int]:
        materials = set()
        for y in range(self.TILE_SIZE):
            for x in range(self.TILE_SIZE):
                vis_data = self.visibility_buffer[tile_y * 8 + y][tile_x * 8 + x]
                materials.add(self._get_material_id(vis_data))
        return materials
```

### ADR-VISBUF-004: Material-Sorted Deferred Texturing

**Context**: Shading pass needs efficient material dispatch.

**Decision**: Sort tiles by material, dispatch shading waves per material.

**Rationale**:
- All pixels in wave have same material (coherent branching)
- Single material bind per wave
- Indirect dispatch for variable tile counts
- Reduces shader permutations

**Implementation**:
```python
def generate_shading_jobs(self) -> dict[int, list[tuple[int, int]]]:
    jobs: dict[int, list] = {}  # material_id -> tile list
    for tile in self.tiles:
        for material_id in tile.materials:
            jobs.setdefault(material_id, []).append(tile.coords)
    return jobs
```

### ADR-VISBUF-005: Depth-Based Position Reconstruction

**Context**: Deferred shading needs world position from visibility buffer.

**Decision**: Store depth in visibility pass, reconstruct position in shading pass.

**Rationale**:
- Avoids storing full position (12 bytes) per pixel
- Depth + screen coords + inverse projection = world position
- Same technique as traditional deferred rendering
- Compatible with TAA/motion vectors

**Implementation**:
```python
def reconstruct_position(self, x: int, y: int, depth: float) -> Vec3:
    ndc_x = (x / width) * 2.0 - 1.0
    ndc_y = (y / height) * 2.0 - 1.0
    ndc = Vec4(ndc_x, ndc_y, depth, 1.0)
    world = self.inverse_vp @ ndc
    return Vec3(world.x / world.w, world.y / world.w, world.z / world.w)
```

## Data Flow

### Input Data

```python
class TriangleInput:
    vertices: list[Vec3]  # Clip-space positions (after MVP)
    instance_id: int
    triangle_id: int
    material_id: int
```

### Visibility Buffer

```python
class VisibilityBuffer:
    data: list[list[int]]   # width x height, packed visibility
    depth: list[list[float]] # width x height, depth values
```

### Shading Output

```python
class ShadingResult:
    color: list[list[Vec4]]  # RGBA per pixel
```

## GPU Port Considerations

### Shader Mapping

| Stage | Shader Type | Invocations |
|-------|-------------|-------------|
| VisibilityBufferPass | Fragment | Per-pixel (rasterization) |
| MaterialTileClassifier | Compute | 1 thread per tile |
| MaterialSortingPass | Compute | Prefix sum + scatter |
| DeferredTexturingPass | Compute | 1 thread per pixel |

### Buffer Layout

```wgsl
struct VisibilityData {
    packed: u32,  // 12-bit tri + 20-bit instance
}

// Visibility buffer as storage texture
@group(0) @binding(0) var visibility: texture_storage_2d<r32uint, write>;
@group(0) @binding(1) var depth: texture_depth_2d;

// Material tile indirect dispatch
struct MaterialDispatch {
    material_id: u32,
    tile_count: u32,
    tile_offset: u32,
}
```

### Atomic Operations

Visibility buffer requires atomic min for depth test:
```wgsl
// Pack depth into upper bits, visibility into lower bits
let packed = (depth_bits << 32) | visibility_data;
atomicMin(&visibility_buffer[pixel], packed);
```

## Dependencies

- Requires meshlet stream from Phase 2
- Requires culled instance list from Phase 1
- Provides visibility data for material shading
- Integrates with lighting system for final output
