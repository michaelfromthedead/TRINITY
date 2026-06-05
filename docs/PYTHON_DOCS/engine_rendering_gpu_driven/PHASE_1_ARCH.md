# PHASE 1 ARCHITECTURE: Culling Pipeline

## Overview

Phase 1 covers the multi-stage GPU culling pipeline implemented in `culling.py` (1,109 lines). This is the foundational filtering stage that reduces the set of renderable objects before any drawing occurs.

## Components

### Core Data Structures

| Type | Purpose |
|------|---------|
| `Vec3`, `Vec4` | 3D/4D vector math with dot, length, normalize |
| `AABB` | Axis-aligned bounding box (min/max corners) |
| `Frustum` | 6 normalized planes for view frustum |
| `CullingConstants` | Epsilon values, screen thresholds |

### Culling Stages

```
Input: List[Instance] with bounds
    |
    v
[FrustumCuller] -- Gribb-Hartmann plane extraction from VP matrix
    |              Tests: sphere-inside-frustum, AABB-inside-frustum
    v
[OcclusionCuller] -- HZB mip pyramid construction
    |                 Screen-space AABB projection
    |                 Conservative depth test against pyramid
    v
[DistanceCuller] -- Camera distance calculation
    |               LOD selection based on distance thresholds
    v
[SmallFeatureCuller] -- Screen-space size estimation
    |                   Sub-pixel rejection
    v
Output: Visibility bitmask + LOD indices
```

## Architecture Decisions

### ADR-GPU-CULL-001: Gribb-Hartmann Plane Extraction

**Context**: Need to extract frustum planes from combined view-projection matrix.

**Decision**: Use Gribb-Hartmann method (add/subtract matrix rows for plane coefficients).

**Rationale**:
- Works with arbitrary projection matrices (perspective, ortho)
- Avoids explicit field-of-view / near-far calculation
- Planes are already in world space

**Implementation** (lines 202-272):
```python
# Left plane: row3 + row0
frustum.planes[LEFT] = Vec4(
    m[0][3] + m[0][0], m[1][3] + m[1][0],
    m[2][3] + m[2][0], m[3][3] + m[3][0]
)
```

### ADR-GPU-CULL-002: HZB Max-Reduction

**Context**: Hierarchical Z-Buffer needs a reduction strategy for mip pyramid.

**Decision**: Use max-reduction (keep maximum depth per 2x2 tile).

**Rationale**:
- With reverse-Z (near=1.0, far=0.0), max preserves closest depth
- Conservative test: if max(tile) < object_depth, object is fully occluded
- Avoids false positives (never culls visible objects)

**Implementation** (lines 539-577):
```python
for y in range(new_height):
    for x in range(new_width):
        max_depth = max(
            current_level[y0][x0], current_level[y0][x1],
            current_level[y1][x0], current_level[y1][x1]
        )
```

### ADR-GPU-CULL-003: Screen-Space AABB for Occlusion

**Context**: Need to project 3D bounds to screen for HZB query.

**Decision**: Project AABB corners to screen, compute 2D bounding rectangle.

**Rationale**:
- AABB projection is conservative (screen rect contains all possible projections)
- Faster than sphere projection (8 points vs iterative refinement)
- Screen rect directly maps to HZB mip level selection

### ADR-GPU-CULL-004: LOD Integration in Distance Culler

**Context**: Distance culling and LOD selection share the same distance metric.

**Decision**: Combine distance culling with LOD selection in single pass.

**Rationale**:
- Avoids redundant distance calculation
- LOD distances are superset of cull distance
- Returns both visibility and LOD index per object

### ADR-GPU-CULL-005: Pipeline Composition

**Context**: Multiple culling stages need to be combined efficiently.

**Decision**: Use `CullingPipeline` class that sequences cullers with early-out.

**Rationale**:
- Each stage receives only survivors from previous stage
- Pipeline order is configurable
- Statistics tracking per stage for profiling

## Data Flow

### Input Data

```python
class InstanceCullData:
    bounding_sphere: (Vec3, float)  # center, radius
    bounding_aabb: AABB
    transform: Mat4x4
    lod_distances: list[float]
```

### Output Data

```python
class CullResult:
    visible: bool
    lod_index: int
    screen_size: float
    distance: float
```

### Pipeline State

```python
class CullingPipelineState:
    frustum: Frustum
    hzb_pyramid: list[list[list[float]]]
    camera_position: Vec3
    screen_dimensions: (int, int)
    lod_bias: float
```

## GPU Port Considerations

### Compute Shader Mapping

| Stage | Thread Mapping | Shared Memory |
|-------|---------------|---------------|
| FrustumCuller | 1 thread per instance | None |
| OcclusionCuller | 1 thread per instance | HZB pyramid in texture |
| DistanceCuller | 1 thread per instance | None |
| SmallFeatureCuller | 1 thread per instance | None |

### Buffer Layout

```wgsl
struct CullInput {
    sphere_center: vec3<f32>,
    sphere_radius: f32,
    aabb_min: vec3<f32>,
    _pad0: f32,
    aabb_max: vec3<f32>,
    _pad1: f32,
}

struct CullOutput {
    visible: u32,
    lod_index: u32,
}
```

### Pipeline Optimization

For GPU execution, consider:
1. **Two-Phase Culling** - Phase 1 (frustum) is cheap, Phase 2 (occlusion) is expensive
2. **Stream Compaction** - Compact visible indices between phases
3. **Prefix Sum** - For generating compacted draw lists

## Dependencies

- Requires VP matrix from camera system
- Requires depth buffer from previous frame (for HZB)
- Provides visibility bitmask to indirect draw system
