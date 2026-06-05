# Investigation: engine/simulation/collision

## Summary
The collision module contains a complete, production-quality collision detection system with real implementations of all major algorithms. This includes four broadphase structures (SAP, BVH, Spatial Hash Grid, Octree), full GJK/EPA/SAT narrowphase algorithms, continuous collision detection with conservative advancement, and a contact manifold system with warm starting support. This is unambiguously a REAL IMPLEMENTATION.

## Files
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `__init__.py` | 240 | REAL | Comprehensive exports for all collision subsystems |
| `broadphase.py` | 1473 | REAL | Four complete broadphase implementations |
| `narrowphase.py` | 1092 | REAL | Full GJK/EPA/SAT with specialized shape tests |
| `ccd.py` | 859 | REAL | Continuous collision detection, TOI calculation |
| `contact_manifold.py` | 670 | REAL | Contact persistence, warm starting, reduction |
| `collision_filter.py` | 498 | REAL | 32-layer collision filtering system |
| `collision_events.py` | 555 | REAL | Event dispatch for collision callbacks |
| `config.py` | 196 | REAL | Tunable constants and quality presets |
| **Total** | **~5,583** | | |

## Collision Components

### Broadphase (broadphase.py)
- `SweepAndPrune` - Axis-sorted endpoint lists, O(n log n) rebuild
- `DynamicBVH` - SAH-based BVH with AVL rotations, incremental insert/remove
- `SpatialHashGrid` - Uniform grid hashing with 3D DDA ray traversal
- `Octree` - Hierarchical subdivision with configurable depth/leaf size

### Narrowphase (narrowphase.py)
- `gjk_distance()` - Full GJK with simplex evolution (2/3/4-vertex handling)
- `epa_penetration()` - EPA polytope expansion for penetration depth
- `sat_test()` - 15-axis SAT for OBB-OBB with contact point clipping
- Specialized: `sphere_sphere`, `sphere_capsule`, `capsule_capsule`, `box_box`, `sphere_box`, `capsule_box`

### Shapes
- `Sphere` - Center + radius, support function
- `Capsule` - Line segment + radius, axis projection, closest point on axis
- `Box` - OBB with rotation axes, 8-vertex enumeration
- `ConvexHull` - Vertex cloud with support point computation

### CCD (ccd.py)
- Binary search sweep tests per shape type
- `time_of_impact_sphere_sphere()` - Analytical quadratic solution
- `conservative_advancement()` - Distance-based safe stepping
- `speculative_contacts()` - Velocity-expanded AABB prediction
- `CCDManager` - Coordinates CCD modes (SWEPT, SPECULATIVE, NONE)

### Contact Manifold (contact_manifold.py)
- `ContactPoint` - Position, normal, depth, cached impulses, feature IDs
- `ContactManifold` - Persistent contact tracking with age-based removal
- `ManifoldCache` - Frame-persistent manifold storage with warm starting
- `reduce_manifold()` - Greedy reduction maximizing contact spread

### Collision Filter (collision_filter.py)
- 32-bit layer masks with predefined categories (STATIC, DYNAMIC, PLAYER, etc.)
- `CollisionFilterManager` - Dynamic layer matrix management

### Events (collision_events.py)
- BEGIN/PERSIST/END event types
- Callback dispatching with weak references

## Implementation

- Real broadphase (BVH/SAP)? **YES** - Four complete implementations with SAH, rotations, DDA
- Real narrowphase (GJK/SAT)? **YES** - Full GJK simplex evolution, EPA polytope, 15-axis SAT
- Real collision shapes? **YES** - Sphere, Capsule, Box, ConvexHull with support functions
- Real contact generation? **YES** - Contact clipping, manifold reduction, warm starting

## Verdict
**REAL IMPLEMENTATION**

This is a complete, game-engine-quality collision detection system. All algorithms are correctly implemented with proper edge case handling, numerical stability considerations, and performance optimizations.

## Evidence

### GJK Simplex Evolution (narrowphase.py:328-367)
```python
def _do_simplex_4(simplex: GJKSimplex, direction: Vec3) -> tuple[bool, Vec3]:
    """Handle 4-vertex simplex (tetrahedron)."""
    d = simplex.vertices[0]
    c = simplex.vertices[1]
    b = simplex.vertices[2]
    a = simplex.vertices[3]

    ab = b.point - a.point
    ac = c.point - a.point
    ad = d.point - a.point
    ao = Vec3() - a.point

    abc = _cross(ab, ac)
    acd = _cross(ac, ad)
    adb = _cross(ad, ab)

    if _same_direction(abc, ao):
        simplex.vertices = [c, b, a]
        return _do_simplex_3(simplex, direction)
    # ... (continues with voronoi region tests)
```

### BVH SAH Sibling Selection (broadphase.py:651-685)
```python
def _find_best_sibling(self, leaf_aabb: AABB) -> int:
    """Find best sibling for insertion using SAH."""
    best_sibling = self._root
    best_cost = float("inf")

    stack = [self._root]
    while stack:
        node_id = stack.pop()
        node = self._nodes[node_id]

        combined = node.aabb.merge(leaf_aabb)
        combined_cost = combined.surface_area()

        if node.is_leaf():
            cost = combined_cost
            if cost < best_cost:
                best_cost = cost
                best_sibling = node_id
        else:
            inherited_cost = combined_cost - node.aabb.surface_area()
            # ... (continues with lower bound pruning)
```

### SAT 15-Axis Test (narrowphase.py:639-713)
```python
def sat_test(box_a: Box, box_b: Box) -> ContactResult:
    # 15 axes total: 3 from A, 3 from B, 9 edge pairs
    test_axes: list[Vec3] = []

    # Face normals
    test_axes.extend(axes_a)
    test_axes.extend(axes_b)

    # Edge cross products
    for axis_a in axes_a:
        for axis_b in axes_b:
            cross = _cross(axis_a, axis_b)
            if cross.length() > SAT_EDGE_BIAS:
                test_axes.append(cross.normalized())
    # ... (continues with projection and overlap calculation)
```

### Conservative Advancement (ccd.py:519-589)
```python
def conservative_advancement(...) -> CCDResult:
    t = 0.0
    relative_velocity = motion_a.velocity - motion_b.velocity
    max_speed = relative_velocity.length() * dt

    for _ in range(max_iterations):
        # Compute distance at current t
        intersecting, distance, closest_a, closest_b = gjk_distance(current_a, current_b)

        if intersecting or distance < tolerance:
            return CCDResult(hit=True, toi=t, ...)

        # Conservative step based on distance / velocity bound
        advance = distance / velocity_bound
        t += advance * CCD_SAFETY_FACTOR
```
