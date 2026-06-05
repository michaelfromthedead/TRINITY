# PHASE 3 ARCHITECTURE: Geometry Completeness

---

## Phase Overview

Complete the stubbed geometry algorithms identified in the investigation:

1. `_compute_hull()` - Convex hull computation (currently returns input points)
2. `_build_bvh()` - Mesh BVH for triangle meshes (currently returns single AABB)
3. Box-box overlap - True SAT test (currently uses conservative AABB approximation)

---

## Architectural Decisions

### AD-3.1: Quickhull for Convex Hull

**Decision**: Implement Quickhull algorithm for convex hull computation.

**Rationale**:
- Average O(n log n), worst O(n^2) - acceptable for mesh authoring (not runtime)
- Robust to degenerate inputs with proper handling
- Well-documented algorithm with clear implementation path

**Alternatives Considered**:
- Gift wrapping: O(n^2) always, simpler but slower
- Incremental: Good for streaming, but we have all points upfront
- Chan's algorithm: Optimal O(n log h), but complex for marginal gain

### AD-3.2: BVH for Mesh Triangles

**Decision**: Reuse BVH implementation from Phase 2 for triangle meshes.

**Rationale**:
- Same SAH construction applies to triangles
- Unified codebase, fewer algorithms to maintain
- Triangle AABB is trivial: min/max of vertices

**Key Difference from Object BVH**:
- Static (mesh doesn't deform): build once, no incremental updates
- Leaf contains triangle indices, not rigid body indices

### AD-3.3: SAT for OBB-OBB

**Decision**: Implement full Separating Axis Theorem for oriented box pairs.

**Rationale**:
- SAT is exact (no false positives or negatives)
- 15 axes to test: 3+3 face normals + 9 edge cross products
- Early exit on first separating axis makes average case fast

**The 15 Axes**:
```
Face normals of A: A.x, A.y, A.z (3)
Face normals of B: B.x, B.y, B.z (3)
Edge cross products: A.i x B.j for i,j in {x,y,z} (9)
```

---

## Component Boundaries

### New Components

| Component | Responsibility |
|-----------|----------------|
| `Quickhull` | Convex hull algorithm |
| `MeshBVH` | BVH specialized for static triangle meshes |
| `SATTest` | Separating Axis Theorem for OBB pairs |

### Modified Components

| Component | Change |
|-----------|--------|
| `collision_shapes.py` | `_compute_hull()` calls Quickhull |
| `collision_shapes.py` | `MeshCollider._build_bvh()` builds MeshBVH |
| `physics_world.py` | Box-box narrowphase uses SAT |

---

## Interfaces

### Quickhull

```python
class Quickhull:
    @staticmethod
    def compute(points: List[Vec3]) -> ConvexHull:
        """
        Compute convex hull of 3D point set.
        
        Returns:
            ConvexHull with vertices, faces (as vertex index triples), 
            and face normals.
        """

@dataclass
class ConvexHull:
    vertices: List[Vec3]
    faces: List[Tuple[int, int, int]]  # Vertex indices, CCW winding
    normals: List[Vec3]  # One per face
```

### MeshBVH

```python
class MeshBVH:
    def __init__(self, vertices: List[Vec3], triangles: List[Tuple[int, int, int]]):
        """Build BVH over triangles."""
        
    def query_ray(self, origin: Vec3, direction: Vec3) -> Optional[RayHit]:
        """Return closest triangle hit."""
        
    def query_aabb(self, aabb: AABB) -> List[int]:
        """Return triangle indices overlapping AABB."""
        
    def query_sphere(self, center: Vec3, radius: float) -> List[int]:
        """Return triangle indices within sphere."""

@dataclass
class RayHit:
    triangle_index: int
    t: float  # Ray parameter
    barycentric: Tuple[float, float, float]
    normal: Vec3
```

### SAT Test

```python
class SATTest:
    @staticmethod
    def obb_obb(box_a: OBB, box_b: OBB) -> bool:
        """
        Test if two oriented bounding boxes overlap.
        
        Returns:
            True if overlapping, False if separated.
        """

@dataclass
class OBB:
    center: Vec3
    half_extents: Vec3  # Half-size along each local axis
    orientation: Mat3   # Local-to-world rotation
```

---

## Algorithmic Details

### Quickhull

```
1. Find extreme points (min/max on each axis) -> initial simplex
2. For each face of simplex:
   - Find farthest point outside face's half-space
   - If found: create cone from point to face edges, remove old face
   - Recurse on new faces
3. Terminate when no points outside any face
```

**Degenerate Handling**:
- Coplanar points: Project to 2D, compute 2D hull, extrude to thin 3D hull
- Collinear points: Return line segment (degenerate hull)
- Single point: Return single-vertex hull

### MeshBVH

```
1. Compute AABB for each triangle
2. Build BVH using SAH (reuse Phase 2 implementation)
3. Store triangle indices in leaves
```

**Ray-Triangle Intersection (Moller-Trumbore)**:
```
edge1 = v1 - v0
edge2 = v2 - v0
h = cross(direction, edge2)
a = dot(edge1, h)
if abs(a) < epsilon: return None  # Parallel
f = 1 / a
s = origin - v0
u = f * dot(s, h)
if u < 0 or u > 1: return None
q = cross(s, edge1)
v = f * dot(direction, q)
if v < 0 or u + v > 1: return None
t = f * dot(edge2, q)
if t > epsilon: return (t, u, v, 1-u-v)
return None
```

### SAT for OBB

```
For each of 15 axes:
    project_a = project_obb(box_a, axis)
    project_b = project_obb(box_b, axis)
    if not overlap_1d(project_a, project_b):
        return False  # Separating axis found
return True  # No separating axis, boxes overlap
```

**Projection onto axis**:
```
def project_obb(box, axis):
    center_proj = dot(box.center, axis)
    # Extent projection: sum of |axis . local_axis| * half_extent
    r = sum(abs(dot(axis, box.orientation.col(i))) * box.half_extents[i] 
            for i in range(3))
    return (center_proj - r, center_proj + r)
```

---

## Performance Considerations

### Quickhull

- Only called during mesh authoring, not runtime
- Acceptable performance: 10,000 points in < 100ms
- Cache result in ConvexHullCollider

### MeshBVH

- Built once per mesh, stored in MeshCollider
- Query performance critical for collision detection
- Target: 1M triangles queryable in < 1ms

### SAT

- Called during narrowphase, must be fast
- Early exit on first separating axis (average ~5 axes tested)
- No heap allocations in hot path

---

## Dependencies

### Internal

- `engine/simulation/physics/collision_shapes.py`
- `engine/simulation/physics/bvh.py` (from Phase 2)
- `engine/simulation/solver/jacobian.py` (Vec3, Mat3)

### External

- None

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Quickhull numerical instability | MEDIUM | Robust predicates, epsilon handling |
| Degenerate hull crashes | HIGH | Explicit degenerate case handling |
| SAT edge-edge axis degeneracy | MEDIUM | Skip near-zero cross products |
| MeshBVH build time for large meshes | LOW | Acceptable for offline build |
