# PHASE 2 TODO: Broadphase Acceleration

---

## T-2.1: Implement BVHNode Data Structure

**File**: `engine/simulation/physics/bvh.py`

**Tasks**:
- [ ] Create `BVHNode` dataclass with AABB, children, object indices
- [ ] Implement `is_leaf` property
- [ ] Implement `depth` method for debugging
- [ ] Implement `count_nodes` method for metrics

**Acceptance Criteria**:
- [ ] Leaf nodes have objects list, no children
- [ ] Internal nodes have children, no objects list
- [ ] Memory footprint documented (bytes per node)

---

## T-2.2: Implement SAH Builder

**File**: `engine/simulation/physics/bvh.py`

**Tasks**:
- [ ] Implement `SAHBuilder` class
- [ ] Compute centroid bounds for object set
- [ ] Find optimal split axis (largest centroid spread)
- [ ] Implement SAH cost function: `SA(left) * N(left) + SA(right) * N(right)`
- [ ] Implement binned SAH for O(n) split evaluation (not O(n^2))
- [ ] Recursive build with leaf termination at max_leaf_size

**Binned SAH Algorithm**:
```
1. Divide axis range into K bins (K=16 typical)
2. Count objects per bin
3. Compute prefix/suffix sums for N and SA
4. Evaluate K-1 split candidates in O(K)
5. Choose minimum cost split
```

**Acceptance Criteria**:
- [ ] Build time is O(n log n) verified by benchmark
- [ ] Tree depth is O(log n) for uniform distribution
- [ ] No crash on degenerate input (all objects at same position)

---

## T-2.3: Implement BVH Tree Container

**File**: `engine/simulation/physics/bvh.py`

**Tasks**:
- [ ] Create `BVH` class wrapping root node
- [ ] Implement constructor taking list of AABBs
- [ ] Implement `query_pairs()` returning overlapping index pairs
- [ ] Implement `query_aabb(aabb)` returning overlapping indices
- [ ] Implement `query_ray(origin, direction)` for future raycast use

**Acceptance Criteria**:
- [ ] `query_pairs()` returns superset of true collisions (false positives OK)
- [ ] No false negatives (no missed collisions)
- [ ] Query results are deterministic

---

## T-2.4: Implement Tree Traversal

**File**: `engine/simulation/physics/bvh.py`

**Tasks**:
- [ ] Implement iterative stack-based traversal (not recursive)
- [ ] Self-intersection traversal for `query_pairs()`
- [ ] Single AABB query traversal for `query_aabb()`
- [ ] Track traversal statistics (nodes visited, leaves checked)

**Self-Intersection Algorithm**:
```
stack = [(root, root)]
pairs = []
while stack:
    a, b = stack.pop()
    if not overlap(a.aabb, b.aabb):
        continue
    if a.is_leaf and b.is_leaf:
        # All pairs from a.objects x b.objects (minus duplicates if a == b)
    else:
        # Push child pairs
```

**Acceptance Criteria**:
- [ ] No stack overflow for 100,000 objects
- [ ] No duplicate pairs in output
- [ ] Self-pairs (i, i) excluded

---

## T-2.5: Implement Incremental Update

**File**: `engine/simulation/physics/bvh.py`

**Tasks**:
- [ ] Implement `update(index, new_aabb)` for single object
- [ ] Track object-to-leaf mapping
- [ ] Implement ancestor refit after leaf update
- [ ] Track refit cost metric (nodes refitted / total nodes)
- [ ] Implement `rebuild_if_needed()` with cost threshold

**Refit Algorithm**:
```
def refit_ancestors(leaf):
    node = leaf.parent
    while node:
        old_aabb = node.aabb
        node.aabb = union(node.left.aabb, node.right.aabb)
        if old_aabb == node.aabb:
            break  # No further propagation needed
        node = node.parent
```

**Acceptance Criteria**:
- [ ] Single update is O(log n) not O(n)
- [ ] Rebuild triggers when refit_cost > 0.3
- [ ] After rebuild, refit_cost resets to 0

---

## T-2.6: Implement BVHBroadphase Adapter

**File**: `engine/simulation/physics/broadphase.py`

**Tasks**:
- [ ] Create `BVHBroadphase` class implementing broadphase interface
- [ ] Cache BVH between frames
- [ ] Rebuild BVH when body count changes significantly (>20%)
- [ ] Update BVH for moved bodies
- [ ] Fall back to O(n^2) for N < 16 (BVH overhead not worth it)

**Interface**:
```python
class BVHBroadphase:
    def find_pairs(self, bodies: List[RigidBody]) -> List[Tuple[int, int]]:
        """Return candidate collision pairs."""
```

**Acceptance Criteria**:
- [ ] Drop-in replacement for existing broadphase
- [ ] No behavior change in physics simulation
- [ ] Measurable speedup for N > 100

---

## T-2.7: Integrate with PhysicsWorld

**File**: `engine/simulation/physics/physics_world.py`

**Tasks**:
- [ ] Add `BroadphaseType` enum: `NAIVE`, `BVH`
- [ ] Add broadphase type parameter to `PhysicsWorld.__init__`
- [ ] Replace inline O(n^2) loop with broadphase call
- [ ] Default to `BVH` for new worlds

**Acceptance Criteria**:
- [ ] Existing tests pass with both broadphase types
- [ ] `NAIVE` mode preserved for debugging/comparison
- [ ] No changes to public PhysicsWorld API

---

## T-2.8: Add Broadphase Tests

**File**: `tests/simulation/physics/test_broadphase.py`

**Tasks**:
- [ ] Test BVH build correctness (all pairs returned)
- [ ] Test incremental update correctness
- [ ] Test rebuild threshold behavior
- [ ] Test edge cases: empty world, single body, two bodies

**Correctness Tests**:
- [ ] Compare BVH pairs to naive O(n^2) pairs (must be superset)
- [ ] Verify no missed collisions over 1000 random configurations

**Acceptance Criteria**:
- [ ] 100% pair coverage vs naive baseline
- [ ] No false negatives in any test configuration

---

## T-2.9: Benchmark Broadphase Performance

**File**: `tests/simulation/benchmarks.py` (extend)

**Tasks**:
- [ ] Benchmark BVH build time for N = 100, 500, 1000, 5000
- [ ] Benchmark BVH query_pairs for same sizes
- [ ] Compare to O(n^2) baseline from Phase 1
- [ ] Profile memory usage (nodes, bytes)

**Expected Results**:
```
N=100:  naive=0.5ms,  bvh_build=0.1ms, bvh_query=0.05ms, total=0.15ms
N=500:  naive=8.3ms,  bvh_build=0.3ms, bvh_query=0.2ms,  total=0.5ms
N=1000: naive=32.1ms, bvh_build=0.6ms, bvh_query=0.4ms,  total=1.0ms
N=5000: naive=800ms,  bvh_build=3ms,   bvh_query=2ms,    total=5ms
```

**Acceptance Criteria**:
- [ ] BVH total time < naive time for N > 50
- [ ] BVH scaling is O(n log n) verified by fit
- [ ] Results documented in `baseline_benchmarks.json`

---

## T-2.10: Add BVH Debug Visualization (Optional)

**File**: `engine/simulation/physics/bvh.py`

**Tasks**:
- [ ] Implement `to_debug_lines()` returning AABB edges for rendering
- [ ] Color-code by depth for tree structure visualization
- [ ] Add tree statistics: depth, leaf count, average leaf size

**Acceptance Criteria**:
- [ ] Debug output is optional, no performance cost when disabled
- [ ] Visualization useful for debugging degenerate trees
