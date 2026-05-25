# PHASE 3 TODO: Geometry Completeness

---

## T-3.1: Implement Quickhull Algorithm

**File**: `engine/simulation/physics/convex_hull.py`

**Tasks**:
- [ ] Create `ConvexHull` dataclass (vertices, faces, normals)
- [ ] Implement initial simplex construction from extreme points
- [ ] Implement horizon detection for point insertion
- [ ] Implement face cone construction
- [ ] Implement iterative farthest-point selection
- [ ] Handle numerical precision with epsilon comparisons

**Algorithm Outline**:
```
1. Find 6 extreme points (min/max per axis)
2. Build initial tetrahedron from 4 non-coplanar extremes
3. For each remaining point:
   - Find visible faces (point in front of face)
   - Find horizon edges (boundary of visible region)
   - Create cone from point to horizon
   - Remove visible faces, add cone faces
4. Return vertices and faces
```

**Acceptance Criteria**:
- [ ] Produces valid convex polyhedron (all faces planar, normals outward)
- [ ] All input points inside or on hull (point-in-hull test)
- [ ] Handles degenerate cases without crash

---

## T-3.2: Handle Quickhull Degenerate Cases

**File**: `engine/simulation/physics/convex_hull.py`

**Tasks**:
- [ ] Detect coplanar input points (all points in same plane)
- [ ] Detect collinear input points (all points on same line)
- [ ] Detect single point input
- [ ] Return appropriate degenerate hull for each case

**Degenerate Hulls**:
- Single point: Hull with 1 vertex, 0 faces
- Collinear: Hull with 2 vertices, 0 faces (line segment)
- Coplanar: Hull with N vertices, 0 3D faces (polygon)

**Acceptance Criteria**:
- [ ] No crash on any of 3 degenerate cases
- [ ] Degenerate hull is usable for collision (point/line/polygon)

---

## T-3.3: Integrate Quickhull with ConvexHullCollider

**File**: `engine/simulation/physics/collision_shapes.py`

**Tasks**:
- [ ] Replace `_compute_hull()` stub with Quickhull call
- [ ] Cache computed hull in collider instance
- [ ] Compute support function from hull vertices
- [ ] Compute inertia tensor from hull faces (tetrahedron decomposition)

**Inertia from Hull**:
```
For each face:
    Form tetrahedron with face and centroid
    Compute tetrahedron inertia contribution
    Apply parallel axis theorem
Sum all contributions
```

**Acceptance Criteria**:
- [ ] ConvexHullCollider.support() returns correct extreme point
- [ ] Inertia tensor is positive definite
- [ ] Hull cached after first computation

---

## T-3.4: Implement MeshBVH

**File**: `engine/simulation/physics/mesh_bvh.py`

**Tasks**:
- [ ] Create `MeshBVH` class
- [ ] Compute AABB for each triangle
- [ ] Reuse SAH builder from Phase 2 for construction
- [ ] Store triangle indices in leaf nodes
- [ ] Implement `query_ray()` with Moller-Trumbore intersection
- [ ] Implement `query_aabb()` for overlap queries
- [ ] Implement `query_sphere()` for proximity queries

**Acceptance Criteria**:
- [ ] Build time < 1 second for 100K triangles
- [ ] Query time < 1ms for 1M triangles
- [ ] No missed triangles in query results

---

## T-3.5: Implement Moller-Trumbore Ray-Triangle Test

**File**: `engine/simulation/physics/mesh_bvh.py`

**Tasks**:
- [ ] Implement ray-triangle intersection function
- [ ] Return hit parameter t, barycentric coordinates, normal
- [ ] Handle degenerate triangles (zero area)
- [ ] Handle ray parallel to triangle

**Algorithm**:
```python
def ray_triangle(origin, direction, v0, v1, v2):
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = cross(direction, edge2)
    a = dot(edge1, h)
    if abs(a) < EPSILON:
        return None  # Parallel or degenerate
    f = 1.0 / a
    s = origin - v0
    u = f * dot(s, h)
    if u < 0.0 or u > 1.0:
        return None
    q = cross(s, edge1)
    v = f * dot(direction, q)
    if v < 0.0 or u + v > 1.0:
        return None
    t = f * dot(edge2, q)
    if t > EPSILON:
        return RayHit(t=t, u=u, v=v, w=1-u-v)
    return None
```

**Acceptance Criteria**:
- [ ] Correct barycentric coordinates (sum to 1)
- [ ] Normal points toward ray origin for front face
- [ ] No false positives or negatives

---

## T-3.6: Integrate MeshBVH with MeshCollider

**File**: `engine/simulation/physics/collision_shapes.py`

**Tasks**:
- [ ] Replace `_build_bvh()` stub with MeshBVH construction
- [ ] Cache BVH in MeshCollider instance
- [ ] Use BVH for raycast queries
- [ ] Use BVH for overlap queries

**Acceptance Criteria**:
- [ ] MeshCollider.raycast() uses BVH (not brute force)
- [ ] Performance scales with log(N) not N
- [ ] BVH built once, reused across frames

---

## T-3.7: Implement OBB-OBB SAT Test

**File**: `engine/simulation/physics/sat.py`

**Tasks**:
- [ ] Create `OBB` dataclass (center, half_extents, orientation)
- [ ] Implement interval projection for OBB onto axis
- [ ] Implement 1D interval overlap test
- [ ] Test 3 face normals of box A
- [ ] Test 3 face normals of box B
- [ ] Test 9 edge cross products (A.i x B.j)
- [ ] Early exit on first separating axis

**Projection Formula**:
```python
def project_obb(obb, axis):
    center_proj = dot(obb.center, axis)
    r = sum(abs(dot(axis, obb.orientation.column(i))) * obb.half_extents[i]
            for i in range(3))
    return (center_proj - r, center_proj + r)
```

**Acceptance Criteria**:
- [ ] No false negatives (overlapping boxes always detected)
- [ ] No false positives (separated boxes never detected)
- [ ] Average early exit at axis 5 for random non-overlapping pairs

---

## T-3.8: Handle SAT Edge Cases

**File**: `engine/simulation/physics/sat.py`

**Tasks**:
- [ ] Skip near-zero cross product axes (parallel edges)
- [ ] Handle box-inside-box case
- [ ] Handle face-to-face contact case
- [ ] Handle edge-to-edge contact case

**Edge Cross Product Degeneracy**:
```python
cross = cross_product(a_axis, b_axis)
if length_squared(cross) < EPSILON_SQ:
    continue  # Axes parallel, skip this test
axis = normalize(cross)
```

**Acceptance Criteria**:
- [ ] Parallel edges do not cause division by zero
- [ ] All contact configurations detected correctly

---

## T-3.9: Integrate SAT with Physics World

**File**: `engine/simulation/physics/physics_world.py`

**Tasks**:
- [ ] Replace conservative AABB box-box test with SAT
- [ ] Convert BoxCollider to OBB for SAT test
- [ ] Keep AABB broadphase (SAT is narrowphase only)

**Acceptance Criteria**:
- [ ] Box-box collision detection is exact (no false positives)
- [ ] Existing tests continue to pass
- [ ] No performance regression for typical scenes

---

## T-3.10: Add Geometry Completeness Tests

**File**: `tests/simulation/physics/test_geometry.py`

**Tasks**:
- [ ] Test Quickhull on cube vertices (8 points -> 6 faces)
- [ ] Test Quickhull on sphere samples (N points -> ~N faces)
- [ ] Test Quickhull degenerate cases (coplanar, collinear, single)
- [ ] Test MeshBVH ray query against brute force
- [ ] Test SAT on known overlapping/separated OBB pairs

**Reference Configurations**:
- Unit cube hull: 8 vertices, 12 triangles (or 6 quads)
- Two overlapping boxes: SAT returns True
- Two separated boxes (gap 0.1 on X): SAT returns False

**Acceptance Criteria**:
- [ ] Quickhull produces correct face count for primitives
- [ ] MeshBVH query matches brute force for 1000 random rays
- [ ] SAT matches AABB test for axis-aligned boxes (regression)
